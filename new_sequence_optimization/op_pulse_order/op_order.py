# ga_optimize_blocks_variable_length.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cupy as cp
from deap import base, creator, tools

import cu_rsc as cr
cr.setup_tables()


# ==============================
# Config
# ==============================

@dataclass
class BlockVarLenGAConfig:
    # GA params
    n_gen: int = 40
    pop_size: int = 64
    cx_prob: float = 0.7
    mut_prob: float = 0.8
    py_seed: int = 123

    # Sequence sources
    seq_list_npy: str = "seq_list.npy"
    seed_seq_npy: str = "seq_correct_gen_18.npy"

    # Reverse engineering tolerance
    match_atol: float = 1e-12
    match_rtol: float = 1e-9

    # Block structure
    fixed_n_blocks: int | None = None   # if None, use reverse-engineered seed count
    min_block_len: int = 1
    max_block_len: int = 16             # fixed buffer length per block
    repeat_min: int = 1
    repeat_max: int = 20

    # Mutation params
    index_mut_prob_per_gene: float = 0.08
    length_mut_prob_per_block: float = 0.25
    repeat_mut_prob_per_block: float = 0.25

    random_reindex_mutation: bool = True
    local_index_step_sigma: float = 2.0

    # Length penalty
    length_penalty_weight: float = 5.0  # raw_score - weight * total_num_pulses

    # Optional extra penalty for very different block lengths vs seed
    block_length_change_penalty_weight: float = 0.0

    # Simulation params
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    outdir: str = "block_varlen_ga_runs"


# ==============================
# Reverse engineering helpers
# ==============================

def rows_to_indices(seq_cap: np.ndarray, seq_list: np.ndarray, atol=1e-12, rtol=1e-9) -> np.ndarray:
    seq_cap = np.asarray(seq_cap, dtype=float)
    seq_list = np.asarray(seq_list, dtype=float)

    indices = []
    for row in seq_cap:
        matches = np.where(np.all(np.isclose(seq_list, row, atol=atol, rtol=rtol), axis=1))[0]
        if len(matches) == 0:
            raise ValueError(f"No match found in seq_list for row:\n{row}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous match for row:\n{row}\nMatches: {matches}")
        indices.append(int(matches[0]))
    return np.array(indices, dtype=int)


def compress_indices_to_blocks(indices: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    Greedy compression into repeated contiguous blocks.
    """
    indices = np.asarray(indices, dtype=int)
    n = len(indices)
    out: List[Tuple[np.ndarray, int]] = []

    i = 0
    while i < n:
        best_pattern = None
        best_repeats = 1
        best_score = 1

        max_pat_len = (n - i) // 2
        for pat_len in range(1, max_pat_len + 1):
            pattern = indices[i:i + pat_len]
            repeats = 1

            while i + (repeats + 1) * pat_len <= n:
                nxt = indices[i + repeats * pat_len : i + (repeats + 1) * pat_len]
                if np.array_equal(nxt, pattern):
                    repeats += 1
                else:
                    break

            score = pat_len * repeats
            if repeats > 1 and score > best_score:
                best_pattern = pattern.copy()
                best_repeats = repeats
                best_score = score

        if best_pattern is not None:
            out.append((best_pattern, best_repeats))
            i += len(best_pattern) * best_repeats
        else:
            out.append((np.array([indices[i]], dtype=int), 1))
            i += 1

    return out


def reverse_engineer_blocks(
    seed_seq: np.ndarray,
    seq_list: np.ndarray,
    atol=1e-12,
    rtol=1e-9,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, int]]]:
    idx = rows_to_indices(seed_seq, seq_list, atol=atol, rtol=rtol)
    block_specs = compress_indices_to_blocks(idx)
    return idx, block_specs


# ==============================
# Fixed-number-of-blocks helpers
# ==============================

def split_or_pad_blocks(
    seed_block_specs: List[Tuple[np.ndarray, int]],
    target_n_blocks: int,
) -> List[Tuple[np.ndarray, int]]:
    """
    Force the seed decomposition to have exactly target_n_blocks.

    Strategy:
    - if too many blocks: merge neighbors from the end
    - if too few blocks: split the longest patterns until count matches
    """
    blocks = [(np.asarray(p, dtype=int).copy(), int(r)) for p, r in seed_block_specs]

    if target_n_blocks <= 0:
        raise ValueError("target_n_blocks must be positive")

    # Merge if too many
    while len(blocks) > target_n_blocks:
        p2, r2 = blocks.pop()
        p1, r1 = blocks.pop()
        if r1 != 1 or r2 != 1:
            merged = (np.tile(p1, r1).tolist() + np.tile(p2, r2).tolist())
            blocks.append((np.array(merged, dtype=int), 1))
        else:
            blocks.append((np.concatenate([p1, p2]), 1))

    # Split if too few
    while len(blocks) < target_n_blocks:
        lens = [len(p) for p, _ in blocks]
        idx = int(np.argmax(lens))
        p, r = blocks[idx]

        full = np.tile(p, r)
        if len(full) <= 1:
            # Cannot split meaningfully, duplicate a trivial block
            blocks.insert(idx + 1, (full.copy(), 1))
            continue

        cut = len(full) // 2
        left = np.array(full[:cut], dtype=int)
        right = np.array(full[cut:], dtype=int)
        if len(left) == 0 or len(right) == 0:
            raise RuntimeError("Unexpected split failure")

        blocks[idx] = (left, 1)
        blocks.insert(idx + 1, (right, 1))

    return blocks


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# ==============================
# Variable-length genome encoding
# ==============================

def encode_seed_to_varlen_genome(
    block_specs: List[Tuple[np.ndarray, int]],
    max_block_len: int,
    min_block_len: int,
    repeat_min: int,
    repeat_max: int,
    fill_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genome layout:
      [lengths (B), repeats (B), pattern_buffer (B * max_block_len)]

    Returns:
      genome, seed_lengths, seed_repeats
    """
    B = len(block_specs)

    lengths = np.zeros(B, dtype=int)
    repeats = np.zeros(B, dtype=int)
    pattern_buf = np.full((B, max_block_len), fill_value, dtype=int)

    for b, (pattern, rep) in enumerate(block_specs):
        pattern = np.asarray(pattern, dtype=int)
        L = len(pattern)
        if L > max_block_len:
            raise ValueError(
                f"Block {b} has length {L}, exceeds max_block_len={max_block_len}. "
                f"Increase max_block_len."
            )

        lengths[b] = clamp_int(L, min_block_len, max_block_len)
        repeats[b] = clamp_int(int(rep), repeat_min, repeat_max)
        pattern_buf[b, :L] = pattern[:L]

        if L < max_block_len:
            pattern_buf[b, L:] = pattern[L - 1] if L > 0 else fill_value

    genome = np.concatenate([
        lengths,
        repeats,
        pattern_buf.reshape(-1),
    ]).astype(int)

    return genome, lengths.copy(), repeats.copy()


def decode_varlen_genome(
    genome: np.ndarray,
    n_blocks: int,
    max_block_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      lengths: (B,)
      repeats: (B,)
      pattern_buf: (B, max_block_len)
    """
    genome = np.asarray(genome, dtype=int)

    n_len = n_blocks
    n_rep = n_blocks
    n_pat = n_blocks * max_block_len
    expected = n_len + n_rep + n_pat

    if genome.size != expected:
        raise ValueError(f"Genome size {genome.size} != expected {expected}")

    lengths = genome[:n_len]
    repeats = genome[n_len:n_len + n_rep]
    pattern_buf = genome[n_len + n_rep:].reshape(n_blocks, max_block_len)

    return lengths, repeats, pattern_buf


def sanitize_genome(
    genome: np.ndarray,
    *,
    n_blocks: int,
    n_lib: int,
    min_block_len: int,
    max_block_len: int,
    repeat_min: int,
    repeat_max: int,
) -> np.ndarray:
    g = np.asarray(genome, dtype=int).copy()
    lengths, repeats, pattern_buf = decode_varlen_genome(g, n_blocks, max_block_len)

    lengths[:] = np.clip(lengths, min_block_len, max_block_len)
    repeats[:] = np.clip(repeats, repeat_min, repeat_max)
    pattern_buf[:] = np.clip(pattern_buf, 0, n_lib - 1)

    out = np.concatenate([
        lengths,
        repeats,
        pattern_buf.reshape(-1),
    ]).astype(int)
    return out


def genome_to_block_specs(
    genome: np.ndarray,
    *,
    n_blocks: int,
    max_block_len: int,
) -> List[Tuple[np.ndarray, int]]:
    lengths, repeats, pattern_buf = decode_varlen_genome(genome, n_blocks, max_block_len)

    specs = []
    for b in range(n_blocks):
        L = int(lengths[b])
        R = int(repeats[b])
        pattern = pattern_buf[b, :L].copy()
        specs.append((pattern, R))
    return specs


def build_sequence_from_block_specs(
    block_specs: List[Tuple[np.ndarray, int]],
    seq_list: np.ndarray
) -> np.ndarray:
    chunks = []
    for pattern, repeats in block_specs:
        pattern = np.asarray(pattern, dtype=int)
        block = seq_list[pattern]
        chunks.append(np.tile(block, (int(repeats), 1)))
    if not chunks:
        return np.zeros((0, seq_list.shape[1]), dtype=float)
    return np.vstack(chunks)


def total_sequence_length(block_specs: List[Tuple[np.ndarray, int]]) -> int:
    return int(sum(len(pattern) * int(rep) for pattern, rep in block_specs))


# ==============================
# Scoring
# ==============================

def score_molecules(
    mol: cp.ndarray,
    *,
    max_nz: int = 1,
) -> Tuple[int, float, float]:
    if mol.ndim != 2 or mol.shape[1] < 6:
        raise ValueError("mol must be shape (N,6)")

    n_x = mol[:, 0]
    n_y = mol[:, 1]
    n_z = mol[:, 2]
    mN  = mol[:, 3]
    spin = mol[:, 4]
    is_lost = mol[:, 5]

    N = int(mol.shape[0])

    survived = (mN == 1) & (spin == 0) & (is_lost == 0)
    surv_count = int(cp.count_nonzero(survived).get())

    meet_n_condition = (n_x == 0) & (n_y == 0) & (n_z <= int(max_nz))
    good_mask = survived & meet_n_condition

    score_count = int(cp.count_nonzero(good_mask).get())
    good_fraction = float(surv_count / N) if N > 0 else 0.0

    if surv_count > 0:
        nz_sum = cp.sum(n_z[survived]).astype(cp.float64)
        nz_bar = float((nz_sum / float(surv_count)).get())
    else:
        nz_bar = float("nan")

    return score_count/N, good_fraction, nz_bar


def score_sequence(
    pulses_host: np.ndarray,
    cfg: BlockVarLenGAConfig,
    res: cr.GPUResources,
) -> Tuple[int, float, float]:
    mol = cr.build_thermal_molecules(int(cfg.n_molecules), list(cfg.temp))

    cr.raman_cool_with_pumping(
        molecules_dev=mol,
        pulses_dev=pulses_host,
        res=res,
        K_max=int(cfg.K_max),
        show_progress=False,
    )

    score, good, n_bar = score_molecules(mol)
    return score, n_bar, good


# ==============================
# GA main
# ==============================

def run_block_varlen_ga(cfg: BlockVarLenGAConfig, res: cr.GPUResources) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.py_seed)

    seq_list = np.load(cfg.seq_list_npy)
    seed_seq = np.load(cfg.seed_seq_npy)

    if seq_list.ndim != 2 or seq_list.shape[1] < 4:
        raise ValueError("seq_list must be shape (N_lib, 4)")
    if seed_seq.ndim != 2 or seed_seq.shape[1] < 4:
        raise ValueError("seed_seq must be shape (P, 4)")

    n_lib = int(seq_list.shape[0])

    # Reverse engineer seed
    seed_indices, seed_block_specs_raw = reverse_engineer_blocks(
        seed_seq,
        seq_list,
        atol=cfg.match_atol,
        rtol=cfg.match_rtol,
    )

    n_blocks = cfg.fixed_n_blocks if cfg.fixed_n_blocks is not None else len(seed_block_specs_raw)
    seed_block_specs = split_or_pad_blocks(seed_block_specs_raw, n_blocks)

    seed_pattern_lengths = [len(p) for p, _ in seed_block_specs]
    seed_total_len = int(sum(len(p) * r for p, r in seed_block_specs))

    if max(seed_pattern_lengths) > cfg.max_block_len:
        raise ValueError(
            f"Seed block length {max(seed_pattern_lengths)} exceeds max_block_len={cfg.max_block_len}. "
            f"Increase max_block_len."
        )

    print(f"Reverse engineered seed into {len(seed_block_specs_raw)} blocks")
    print(f"Using fixed_n_blocks = {n_blocks}")
    print(f"Seed block lengths = {seed_pattern_lengths}")
    print(f"Seed total pulse count = {seed_total_len}")

    seed_genome, seed_lengths, seed_repeats = encode_seed_to_varlen_genome(
        seed_block_specs,
        max_block_len=cfg.max_block_len,
        min_block_len=cfg.min_block_len,
        repeat_min=cfg.repeat_min,
        repeat_max=cfg.repeat_max,
        fill_value=0,
    )

    # DEAP setup
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()

    def init_ind() -> creator.Individual:
        g = seed_genome.copy()

        lengths, repeats, pattern_buf = decode_varlen_genome(g, n_blocks, cfg.max_block_len)

        # mutate active pattern entries lightly around seed
        for b in range(n_blocks):
            L = int(lengths[b])
            for j in range(L):
                if rng.random() < cfg.index_mut_prob_per_gene:
                    if cfg.random_reindex_mutation:
                        pattern_buf[b, j] = int(rng.integers(0, n_lib))
                    else:
                        step = int(np.round(rng.normal(0.0, cfg.local_index_step_sigma)))
                        pattern_buf[b, j] = clamp_int(pattern_buf[b, j] + step, 0, n_lib - 1)

        # mutate block lengths
        for b in range(n_blocks):
            if rng.random() < cfg.length_mut_prob_per_block:
                step = int(rng.integers(-2, 3))
                lengths[b] = clamp_int(lengths[b] + step, cfg.min_block_len, cfg.max_block_len)

        # mutate repeats
        for b in range(n_blocks):
            if rng.random() < cfg.repeat_mut_prob_per_block:
                step = int(rng.integers(-2, 3))
                repeats[b] = clamp_int(repeats[b] + step, cfg.repeat_min, cfg.repeat_max)

        g = np.concatenate([lengths, repeats, pattern_buf.reshape(-1)]).astype(int)
        g = sanitize_genome(
            g,
            n_blocks=n_blocks,
            n_lib=n_lib,
            min_block_len=cfg.min_block_len,
            max_block_len=cfg.max_block_len,
            repeat_min=cfg.repeat_min,
            repeat_max=cfg.repeat_max,
        )
        return creator.Individual(g)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def genome_to_sequence(ind: np.ndarray) -> Tuple[List[Tuple[np.ndarray, int]], np.ndarray]:
        specs = genome_to_block_specs(
            np.asarray(ind, dtype=int),
            n_blocks=n_blocks,
            max_block_len=cfg.max_block_len,
        )
        seq = build_sequence_from_block_specs(specs, seq_list)
        return specs, seq

    def evaluate(ind: np.ndarray) -> tuple:
        specs, pulses_host = genome_to_sequence(ind)
        num_pulses = int(pulses_host.shape[0])

        raw_score, n_bar, good = score_sequence(pulses_host, cfg, res)

        penalized = float(raw_score) - float(cfg.length_penalty_weight) * float(num_pulses)

        if cfg.block_length_change_penalty_weight > 0.0:
            cur_lengths = np.array([len(p) for p, _ in specs], dtype=float)
            ref_lengths = np.array(seed_pattern_lengths, dtype=float)
            penalized -= float(cfg.block_length_change_penalty_weight) * float(np.abs(cur_lengths - ref_lengths).sum())

        ind.raw_score = float(raw_score)
        ind.penalized_score = float(penalized)
        ind.n_bar = float(n_bar)
        ind.good = float(good)
        ind._num_pulses = int(num_pulses)
        ind.block_specs = [(p.copy(), int(r)) for p, r in specs]

        return (penalized,)

    toolbox.register("evaluate", evaluate)

    def mate(a: np.ndarray, b: np.ndarray):
        """
        Block-aware crossover:
        - swap lengths blockwise
        - swap repeats blockwise
        - swap pattern buffers blockwise
        """
        g1 = np.asarray(a, dtype=int).copy()
        g2 = np.asarray(b, dtype=int).copy()

        L1, R1, P1 = decode_varlen_genome(g1, n_blocks, cfg.max_block_len)
        L2, R2, P2 = decode_varlen_genome(g2, n_blocks, cfg.max_block_len)

        for blk in range(n_blocks):
            if rng.random() < 0.5:
                L1[blk], L2[blk] = L2[blk], L1[blk]
            if rng.random() < 0.5:
                R1[blk], R2[blk] = R2[blk], R1[blk]
            if rng.random() < 0.5:
                tmp = P1[blk].copy()
                P1[blk] = P2[blk]
                P2[blk] = tmp

        child1 = np.concatenate([L1, R1, P1.reshape(-1)]).astype(int)
        child2 = np.concatenate([L2, R2, P2.reshape(-1)]).astype(int)

        child1 = sanitize_genome(
            child1,
            n_blocks=n_blocks,
            n_lib=n_lib,
            min_block_len=cfg.min_block_len,
            max_block_len=cfg.max_block_len,
            repeat_min=cfg.repeat_min,
            repeat_max=cfg.repeat_max,
        )
        child2 = sanitize_genome(
            child2,
            n_blocks=n_blocks,
            n_lib=n_lib,
            min_block_len=cfg.min_block_len,
            max_block_len=cfg.max_block_len,
            repeat_min=cfg.repeat_min,
            repeat_max=cfg.repeat_max,
        )

        a[:] = child1
        b[:] = child2
        return a, b

    toolbox.register("mate", mate)

    def mutate(ind: np.ndarray):
        g = np.asarray(ind, dtype=int).copy()
        lengths, repeats, pattern_buf = decode_varlen_genome(g, n_blocks, cfg.max_block_len)

        # mutate lengths
        for b in range(n_blocks):
            if rng.random() < cfg.length_mut_prob_per_block:
                step = int(rng.integers(-2, 3))
                old_L = int(lengths[b])
                new_L = clamp_int(old_L + step, cfg.min_block_len, cfg.max_block_len)

                # If length expands, initialize new active entries sensibly
                if new_L > old_L:
                    fill_source = pattern_buf[b, old_L - 1] if old_L > 0 else int(rng.integers(0, n_lib))
                    for j in range(old_L, new_L):
                        if cfg.random_reindex_mutation:
                            pattern_buf[b, j] = fill_source
                        else:
                            pattern_buf[b, j] = fill_source

                lengths[b] = new_L

        # mutate active indices only
        for b in range(n_blocks):
            L = int(lengths[b])
            for j in range(L):
                if rng.random() < cfg.index_mut_prob_per_gene:
                    if cfg.random_reindex_mutation:
                        pattern_buf[b, j] = int(rng.integers(0, n_lib))
                    else:
                        step = int(np.round(rng.normal(0.0, cfg.local_index_step_sigma)))
                        pattern_buf[b, j] = clamp_int(pattern_buf[b, j] + step, 0, n_lib - 1)

        # mutate repeats
        for b in range(n_blocks):
            if rng.random() < cfg.repeat_mut_prob_per_block:
                step = int(rng.integers(-2, 3))
                repeats[b] = clamp_int(repeats[b] + step, cfg.repeat_min, cfg.repeat_max)

        g = np.concatenate([lengths, repeats, pattern_buf.reshape(-1)]).astype(int)
        g = sanitize_genome(
            g,
            n_blocks=n_blocks,
            n_lib=n_lib,
            min_block_len=cfg.min_block_len,
            max_block_len=cfg.max_block_len,
            repeat_min=cfg.repeat_min,
            repeat_max=cfg.repeat_max,
        )

        ind[:] = g
        return (ind,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # population
    pop = toolbox.population(n=cfg.pop_size)

    # exact seed as baseline
    baseline_ind = creator.Individual(seed_genome.copy())
    pop[0] = baseline_ind

    hof = tools.HallOfFame(5, similar=lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)))

    history = {
        "gen": [],
        "best": [],
        "avg": [],
        "std": [],
        "min": [],
        "max": [],
        "best_raw_score": [],
        "best_num_pulses": [],
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"BlockVarLenGA_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_info = {
        "seed_seq_npy": cfg.seed_seq_npy,
        "seq_list_npy": cfg.seq_list_npy,
        "seed_indices": seed_indices.tolist(),
        "seed_block_specs_raw": [
            {"pattern": p.tolist(), "repeats": int(r)}
            for p, r in seed_block_specs_raw
        ],
        "seed_block_specs_used": [
            {"pattern": p.tolist(), "repeats": int(r)}
            for p, r in seed_block_specs
        ],
        "n_blocks": n_blocks,
        "seed_pattern_lengths": seed_pattern_lengths,
        "seed_total_len": seed_total_len,
    }
    with open(run_dir / "reverse_engineered_seed.json", "w") as f:
        json.dump(seed_info, f, indent=2)

    # GA loop
    for gen in range(cfg.n_gen):
        print(f"Running BlockVarLen GA generation {gen}")

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best_idx = int(np.argmax(fits))
        best_ind = pop[best_idx]

        best = float(fits.max())
        avg = float(fits.mean())
        std = float(fits.std(ddof=1)) if fits.size > 1 else 0.0

        history["gen"].append(gen)
        history["best"].append(best)
        history["avg"].append(avg)
        history["std"].append(std)
        history["min"].append(float(fits.min()))
        history["max"].append(float(fits.max()))
        history["best_raw_score"].append(float(getattr(best_ind, "raw_score", np.nan)))
        history["best_num_pulses"].append(int(getattr(best_ind, "_num_pulses", -1)))

        print(
            f"[Gen {gen:03d}] "
            f"best_penalized={best:.1f} "
            f"raw={getattr(best_ind, 'raw_score', np.nan):.1f} "
            f"len={getattr(best_ind, '_num_pulses', -1)} "
            f"avg={avg:.1f} std={std:.1f}"
        )

        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        np.save(gen_dir / "best_genome.npy", np.asarray(best_ind, dtype=int))

        best_specs, best_seq = genome_to_sequence(best_ind)
        np.save(gen_dir / "best_sequence.npy", best_seq)

        meta = {
            "gen": gen,
            "penalized_score": float(best_ind.fitness.values[0]),
            "raw_score": getattr(best_ind, "raw_score", None),
            "n_bar": getattr(best_ind, "n_bar", None),
            "good": getattr(best_ind, "good", None),
            "num_pulses": int(best_seq.shape[0]),
            "length_penalty_weight": cfg.length_penalty_weight,
            "block_specs": [
                {"pattern": p.tolist(), "repeats": int(r)}
                for p, r in best_specs
            ],
            "block_lengths": [len(p) for p, _ in best_specs],
            "block_repeats": [int(r) for _, r in best_specs],
        }
        with open(gen_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        hof.update(pop)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for i in range(0, len(offspring) - 1, 2):
            if rng.random() < cfg.cx_prob:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):
            if rng.random() < cfg.mut_prob:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        pop[:] = offspring

    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    np.save(run_dir / "history_gen.npy", np.array(history["gen"]))
    np.save(run_dir / "history_best.npy", np.array(history["best"]))
    np.save(run_dir / "history_avg.npy", np.array(history["avg"]))
    np.save(run_dir / "history_std.npy", np.array(history["std"]))
    np.save(run_dir / "history_min.npy", np.array(history["min"]))
    np.save(run_dir / "history_max.npy", np.array(history["max"]))
    np.save(run_dir / "history_best_raw_score.npy", np.array(history["best_raw_score"]))
    np.save(run_dir / "history_best_num_pulses.npy", np.array(history["best_num_pulses"]))

    final_dir = run_dir / "final_top5"
    final_dir.mkdir(parents=True, exist_ok=True)

    for rank, ind in enumerate(hof, 1):
        specs, seq = genome_to_sequence(ind)
        np.save(final_dir / f"top{rank}_genome.npy", np.asarray(ind, dtype=int))
        np.save(final_dir / f"top{rank}_sequence.npy", seq)

        meta = {
            "rank": rank,
            "penalized_score": float(ind.fitness.values[0]),
            "raw_score": getattr(ind, "raw_score", None),
            "n_bar": getattr(ind, "n_bar", None),
            "good": getattr(ind, "good", None),
            "num_pulses": int(seq.shape[0]),
            "length_penalty_weight": cfg.length_penalty_weight,
            "block_specs": [
                {"pattern": p.tolist(), "repeats": int(r)}
                for p, r in specs
            ],
            "block_lengths": [len(p) for p, _ in specs],
            "block_repeats": [int(r) for _, r in specs],
        }

        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nSaved BlockVarLen GA run to: {run_dir}")


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    M_dev = cr.load_m_table_device()
    res = cr.resources_from_config(M_dev)

    cfg = BlockVarLenGAConfig(
        n_gen=40,
        pop_size=64,
        cx_prob=0.7,
        mut_prob=0.8,
        py_seed=42,

        seq_list_npy="seq_list_0406.npy",
        seed_seq_npy="seq_rebuilt_0406.npy",

        fixed_n_blocks=None,   # use reverse-engineered count
        min_block_len=1,
        max_block_len=16,      # set >= longest seed block you expect
        repeat_min=1,
        repeat_max=20,

        index_mut_prob_per_gene=0.08,
        length_mut_prob_per_block=0.25,
        repeat_mut_prob_per_block=0.25,

        random_reindex_mutation=True,
        local_index_step_sigma=2.0,

        length_penalty_weight=0.001,
        block_length_change_penalty_weight=0.0,

        n_molecules=50_000,
        temp=(25e-6, 25e-6, 25e-6),
        K_max=30,

        outdir="block_varlen_ga_runs",
    )

    run_block_varlen_ga(cfg, res)