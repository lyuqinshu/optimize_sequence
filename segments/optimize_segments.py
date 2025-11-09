# ga_optimize_segments_indices.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cupy as cp
from deap import base, creator, tools

# Project imports
import cu_rsc as cr
cr.setup_tables()
from cu_rsc.build_sequence import pulse_time
from tqdm import tqdm


# ==============================
# Config
# ==============================
@dataclass
class GAConfig:
    # GA
    n_gen: int = 25
    pop_size: int = 32
    cx_prob: float = 0.7
    mut_prob: float = 0.6
    py_seed: int = 123

    # Segments (fixed at 5 as requested)
    num_segments: int = 5
    seg_min_len: int = 1
    seg_max_len: int = 64
    rep_min: int = 1
    rep_max: int = 50

    # Mutate ind (per segment)
    p_add_gene: float = 0.20
    p_drop_gene: float = 0.40
    p_replace_gene: float = 0.40

    # Mutate rep (per segment)
    p_inc_rep: float = 0.35
    p_dec_rep: float = 0.35

    # Crossover: swap segments as units
    p_cx_swap_segment: float = 0.7

    # Penalty (percentage units)
    # Effective penalty = total_pulses * length_penalty * n_molecules
    length_penalty: float = 0.0  # e.g., 0.001 means 0.1% of n_molecules per pulse

    # Simulation
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    # Allowed pool
    allowed_pulses: Optional[List[Tuple[int, int]]] = None
    config_json: Optional[str] = "config.json"

    # Seeding from original sequence blocks
    use_original_segments: bool = True
    seed_repeats: Optional[List[int]] = None  # default [10,5,5,10,10]


# ==============================
# Helpers
# ==============================
def load_allowed_pulses_from_cfg(cfg: GAConfig) -> List[Tuple[int,int]]:
    if cfg.allowed_pulses is not None:
        return [(int(a), int(dn)) for a, dn in cfg.allowed_pulses]
    if cfg.config_json is None:
        raise ValueError("allowed_pulses must be provided or config_json must be set.")
    with open(cfg.config_json, "r") as f:
        j = json.load(f)
    arr = j.get("allowed_pulses", None)
    if not arr:
        raise ValueError("config.json missing 'allowed_pulses'.")
    return [(int(x[0]), int(x[1])) for x in arr]


Segment = Dict[str, Any]  # {"rep": int, "ind": List[int]}


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def build_sequence_from_segments(segments: List[Segment], pool: List[Tuple[int,int]]) -> cp.ndarray:
    """(ind, rep) x 5 → full (P,3) GPU array; pulse time comes from pulse_time(axis,dn)."""
    if not segments:
        return cp.zeros((0, 3), dtype=cp.float64)

    chunks = []
    for seg in segments:
        rep = int(seg["rep"])
        inds: List[int] = [int(x) for x in seg["ind"]]
        if rep <= 0 or len(inds) == 0:
            continue
        axes, dns, times = [], [], []
        for idx in inds:
            ax, dn = pool[idx]
            axes.append(ax)
            dns.append(dn)
            times.append(float(pulse_time(ax, dn)))
        arr = np.column_stack([np.array(axes, float), np.array(dns, float), np.array(times, float)])
        arr = np.tile(arr, (rep, 1))
        chunks.append(arr)

    if not chunks:
        return cp.zeros((0, 3), dtype=cp.float64)

    host = np.concatenate(chunks, axis=0)
    return cp.asarray(host, dtype=cp.float64)


def total_pulses_in_ind(segments: List[Segment]) -> int:
    return int(sum(int(seg["rep"]) * len(seg["ind"]) for seg in segments))


def segment_from_block(block_gpu: cp.ndarray,
                       rep: int,
                       pool: List[Tuple[int,int]],
                       cfg: GAConfig) -> Segment:
    """Seed a segment from an original block (do NOT tile; put tiling in rep)."""
    block = cp.asnumpy(block_gpu)
    if block.ndim != 2 or block.shape[1] < 2:
        raise ValueError("original block must be (P,3) or (P,>=2)")
    P = block.shape[0]
    P = clamp(int(P), cfg.seg_min_len, cfg.seg_max_len)

    # map (axis,dn) to pool indices
    index_map = {p: i for i, p in enumerate(pool)}
    ind: List[int] = []
    for k in range(P):
        pair = (int(block[k, 0]), int(block[k, 1]))
        if pair not in index_map:
            raise ValueError(f"Seed pulse {pair} not present in allowed_pulses.")
        ind.append(int(index_map[pair]))

    rep = clamp(int(rep), cfg.rep_min, cfg.rep_max)
    return {"rep": rep, "ind": ind}


def random_segment(pool_size: int, cfg: GAConfig, rng: np.random.Generator) -> Segment:
    L = int(rng.integers(cfg.seg_min_len, cfg.seg_max_len + 1))
    rep = int(rng.integers(cfg.rep_min, cfg.rep_max + 1))
    ind = rng.integers(0, pool_size, size=L).tolist()
    return {"rep": rep, "ind": [int(x) for x in ind]}


def score_sequence(seq_gpu: cp.ndarray, cfg: GAConfig, res: cr.GPUResources) -> Tuple[float, int]:
    """Return (penalized_score, raw_survivors_zero_state)."""
    mol = cr.build_thermal_molecules_gpu(int(cfg.n_molecules), list(cfg.temp))
    cr.raman_cool_with_pumping(mol, seq_gpu, res, K_max=int(cfg.K_max), show_progress=True)

    n_x, n_y, n_z, is_lost, spin, mN = mol[:, 0], mol[:, 1], mol[:, 2], mol[:, 5], mol[:,4], mol[:,3]
    mask = (is_lost == 0) & (n_x == 0) & (n_y == 0) & (n_z == 0) & (spin == 0) & (mN == 1)
    raw = int(cp.count_nonzero(mask).get())

    # Length penalty in percentage units (per pulse × n_molecules)
    total_pulses = int(seq_gpu.shape[0])
    penalty = total_pulses * cfg.length_penalty * int(cfg.n_molecules)
    penalized = float(raw) - float(penalty)
    return float(penalized), raw


# ==============================
# GA main
# ==============================
def run_ga(cfg: GAConfig, outdir: Path, res: cr.GPUResources) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Pool
    pool = load_allowed_pulses_from_cfg(cfg)
    pool_size = len(pool)
    if pool_size == 0:
        raise ValueError("allowed_pulses is empty.")

    rng = np.random.default_rng(cfg.py_seed)

    # Seed (gen-0 inoculation) from original blocks
    if cfg.use_original_segments:
        original_gpu = cr.get_original_sequences_gpu()  # tuple/list of cp.ndarray blocks
        if len(original_gpu) < cfg.num_segments:
            raise ValueError("Not enough original blocks to seed num_segments.")
        seed_repeats = cfg.seed_repeats or [10, 5, 5, 10, 10]
        seed_repeats = seed_repeats[:cfg.num_segments]
        seed_segments = [
            segment_from_block(original_gpu[i], seed_repeats[i], pool, cfg)
            for i in range(cfg.num_segments)
        ]
    else:
        seed_segments = [random_segment(pool_size, cfg, rng) for _ in range(cfg.num_segments)]

    # DEAP setup
    try: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception: pass
    try: creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception: pass

    toolbox = base.Toolbox()

    def clone_segment(seg: Segment) -> Segment:
        return {"rep": int(seg["rep"]), "ind": [int(x) for x in seg["ind"]]}

    def init_ind() -> creator.Individual:
        segs = [clone_segment(s) for s in seed_segments]
        # light randomization for diversity (mutate one segment slightly)
        if rng.random() < 0.75:
            mutate_one_segment_inplace(segs, pool_size, cfg, rng, force_change=True)
        return creator.Individual(segs)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: List[Segment]) -> tuple:
        seq_gpu = build_sequence_from_segments(ind, pool)
        score, raw = score_sequence(seq_gpu, cfg, res)
        ind._raw_survivors = raw
        ind._num_pulses = int(seq_gpu.shape[0])
        ind._total_pulses_formula = total_pulses_in_ind(ind)
        return (score,)

    toolbox.register("evaluate", evaluate)

    # ---- Variation operators ----
    def crossover(a: List[Segment], b: List[Segment]) -> Tuple[List[Segment], List[Segment]]:
        # swap whole segments
        for i in range(cfg.num_segments):
            if rng.random() < cfg.p_cx_swap_segment:
                a[i], b[i] = b[i], a[i]
        return a, b

    toolbox.register("mate", crossover)

    def mutate_one_segment_inplace(ind: List[Segment],
                                   pool_size: int,
                                   cfg: GAConfig,
                                   rng: np.random.Generator,
                                   force_change: bool = False) -> None:
        sidx = int(rng.integers(0, cfg.num_segments))
        seg = ind[sidx]
        changed = False

        # mutate rep
        if rng.random() < cfg.p_inc_rep:
            new_rep = clamp(seg["rep"] + 1, cfg.rep_min, cfg.rep_max)
            if new_rep != seg["rep"]:
                seg["rep"] = new_rep
                changed = True
        if rng.random() < cfg.p_dec_rep:
            new_rep = clamp(seg["rep"] - 1, cfg.rep_min, cfg.rep_max)
            if new_rep != seg["rep"]:
                seg["rep"] = new_rep
                changed = True

        # mutate ind
        L = len(seg["ind"])
        # add
        if rng.random() < cfg.p_add_gene and L < cfg.seg_max_len:
            pos = int(rng.integers(0, L + 1))
            seg["ind"].insert(pos, int(rng.integers(0, pool_size)))
            changed = True
            L += 1
        # drop
        if rng.random() < cfg.p_drop_gene and L > cfg.seg_min_len:
            pos = int(rng.integers(0, L))
            del seg["ind"][pos]
            changed = True
            L -= 1
        # replace
        if rng.random() < cfg.p_replace_gene and L > 0:
            pos = int(rng.integers(0, L))
            seg["ind"][pos] = int(rng.integers(0, pool_size))
            changed = True

        if not changed and force_change:
            # ensure at least one change
            if len(seg["ind"]) == 0:
                seg["ind"].append(int(rng.integers(0, pool_size)))
            else:
                pos = int(rng.integers(0, len(seg["ind"])))
                seg["ind"][pos] = int(rng.integers(0, pool_size))

    def mutate(ind: List[Segment]):
        mutate_one_segment_inplace(ind, pool_size, cfg, rng, force_change=True)
        return (ind,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Init population / HOF / logs
    pop = toolbox.population(n=cfg.pop_size)

    # Force first individual to exact seed (no randomization)
    seed_ind = creator.Individual([{"rep": int(s["rep"]), "ind": [int(x) for x in s["ind"]]} for s in seed_segments])
    pop[0] = seed_ind

    hof = tools.HallOfFame(
        5,
        similar=lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    )
    history = {"gen": [], "best": [], "avg": [], "std": [], "min": [], "max": []}

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"SEGIDX_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GA loop
    for gen in range(cfg.n_gen):
        print(f"Running generation {gen}")
        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            with tqdm(total=len(invalid), desc=f"Evaluating Gen {gen:03d}", leave=False, dynamic_ncols=True) as pbar:
                for ind in invalid:
                    ind.fitness.values = toolbox.evaluate(ind)
                    pbar.update(1)

        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best, avg = float(fits.max()), float(fits.mean())
        std = float(fits.std(ddof=1)) if fits.size > 1 else 0.0
        history["gen"].append(gen)
        history["best"].append(best)
        history["avg"].append(avg)
        history["std"].append(std)
        history["min"].append(float(fits.min()))
        history["max"].append(float(fits.max()))
        print(f"[Gen {gen:03d}] best={best:.1f}  avg={avg:.1f}  std={std:.1f}")

        # Save top-5 of this generation
        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)[:5]
        for rank, ind in enumerate(ranked, 1):
            seq_gpu = build_sequence_from_segments(ind, pool)
            seq_host = cp.asnumpy(seq_gpu)
            np.save(gen_dir / f"top{rank}_sequence.npy", seq_host)

            meta = {
                "rank": rank,
                "fitness_penalized": ind.fitness.values[0],
                "raw_survivors": getattr(ind, "_raw_survivors", None),
                "num_pulses": getattr(ind, "_num_pulses", int(seq_gpu.shape[0])),
                "total_pulses_formula": total_pulses_in_ind(ind),
                "length_penalty": cfg.length_penalty,
                "segments": [
                    {"rep": int(seg["rep"]),
                     "ind": [int(x) for x in seg["ind"]],
                     "pairs": [list(pool[i]) for i in seg["ind"]]}
                    for seg in ind
                ],
            }
            with open(gen_dir / f"top{rank}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # Update HOF
        hof.update(pop)

        # Selection → Variation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if rng.random() < cfg.cx_prob:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if rng.random() < cfg.mut_prob:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        pop[:] = offspring

    # Final eval & HOF
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)

    # Save run config + history
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    np.save(run_dir / "history_gen.npy",  np.array(history["gen"]))
    np.save(run_dir / "history_best.npy", np.array(history["best"]))
    np.save(run_dir / "history_avg.npy",  np.array(history["avg"]))
    np.save(run_dir / "history_std.npy",  np.array(history["std"]))
    np.save(run_dir / "history_min.npy",  np.array(history["min"]))
    np.save(run_dir / "history_max.npy",  np.array(history["max"]))

    # Save final HoF top-5
    final_dir = run_dir / "final_top5"
    final_dir.mkdir(parents=True, exist_ok=True)
    for rank, ind in enumerate(hof, 1):
        seq_gpu = build_sequence_from_segments(ind, pool)
        seq_host = cp.asnumpy(seq_gpu)
        np.save(final_dir / f"top{rank}_sequence.npy", seq_host)
        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "fitness_penalized": ind.fitness.values[0],
                    "raw_survivors": getattr(ind, "_raw_survivors", None),
                    "num_pulses": int(seq_gpu.shape[0]),
                    "total_pulses_formula": total_pulses_in_ind(ind),
                    "length_penalty": cfg.length_penalty,
                    "segments": [
                        {"rep": int(seg["rep"]),
                         "ind": [int(x) for x in seg["ind"]],
                         "pairs": [list(pool[i]) for i in seg["ind"]]}
                        for seg in ind
                    ],
                },
                f,
                indent=2,
            )
    print(f"\nSaved run to: {run_dir}")


# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    M_dev = cr.load_m_table_device()               # upload to GPU
    res   = cr.resources_from_config(M_dev)
    outdir = Path("ga_runs")

    cfg = GAConfig(
        n_gen=40, pop_size=100, cx_prob=0.7, mut_prob=0.6, py_seed=123,
        num_segments=5, seg_min_len=1, seg_max_len=64, rep_min=1, rep_max=20,
        # Penalty as percentage unit (e.g., 0.001 = 0.1% of n_molecules per pulse)
        length_penalty=0.001,
        n_molecules=50_000, temp=(25e-6, 25e-6, 25e-6), K_max=30,
        allowed_pulses=[(0,-6),(0,-5),(0,-4),(0,-3),(0,-2),(0,-1),
                        (1,-6),(1,-5),(1,-4),(1,-3),(1,-2),(1,-1),
                        (2,-9),(2,-8),(2,-7),(2,-6),(2,-5),(2,-4),(2,-3),(2,-2),(2,-1)],
        use_original_segments=True,
        seed_repeats=[10,5,5,10,10],
        p_inc_rep=0.25, p_dec_rep=0.30,
        p_add_gene=0.20, p_drop_gene=0.40, p_replace_gene=0.40,
    )
    run_ga(cfg, outdir, res)
