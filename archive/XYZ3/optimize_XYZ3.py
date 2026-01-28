# ga_optimize_allowed_pool_seedfile.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
from importlib.resources import files

import numpy as np
import cupy as cp
from deap import base, creator, tools

# Project imports (adjust paths/names to your project)
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

    # Genome length controls (per pulse)
    min_len: int = 20
    max_len: int = 400
    p_add_gene: float = 0.20
    p_drop_gene: float = 0.20
    p_replace_gene: float = 0.40

    # Penalty
    length_penalty: float = 0.0  # score -= len(genome) * length_penalty * len(mol)

    # Simulation
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    # Seed & pool
    seed_file: str = "XYZ2_original.npy"   # saved via: np.save("XY_original.npy", cp.asarray(seq_gpu))
    allowed_pulses: Optional[List[Tuple[int, int]]] = None  # e.g. [(0,-2),(0,-1),(1,-2),...]
    config_json: Optional[str] = "config.json"              # fallback source for allowed_pulses (optional)


# ==============================
# Helpers
# ==============================
def load_allowed_pulses_from_cfg(cfg: GAConfig) -> List[Tuple[int,int]]:
    if cfg.allowed_pulses is not None:
        return [(int(a), int(dn)) for a, dn in cfg.allowed_pulses]
    if cfg.config_json is None:
        raise ValueError("allowed_pulses must be provided in GAConfig or config_json must be set.")
    with open(cfg.config_json, "r") as f:
        j = json.load(f)
    arr = j.get("allowed_pulses", None)
    if not arr:
        raise ValueError("config.json missing 'allowed_pulses'. Expected [[axis, delta_n], ...].")
    out = []
    for item in arr:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError("Each allowed pulse must be [axis, delta_n].")
        out.append((int(item[0]), int(item[1])))
    return out


def load_seed_genome_from_file(seed_path: Path, pool: List[Tuple[int,int]]) -> List[int]:
    """Load seed XY sequence saved with np.save(..., cp.asarray(seq_gpu)); map to pool indices."""
    seq = np.load(seed_path)  # shape (P,3), columns: axis, delta_n, time
    if seq.ndim != 2 or seq.shape[1] < 2:
        raise ValueError(f"Seed file {seed_path} must be (P,3) or (P,>=2).")
    axes = seq[:, 0].astype(int)
    dns  = seq[:, 1].astype(int)
    pairs = [(int(a), int(dn)) for a, dn in zip(axes, dns)]
    index_map = {p: i for i, p in enumerate(pool)}
    genome = []
    for p in pairs:
        if p not in index_map:
            raise ValueError(f"Seed pulse {p} not present in allowed_pulses.")
        genome.append(index_map[p])
    return genome


def build_sequence_from_indices(genome: List[int], pool: List[Tuple[int,int]]) -> cp.ndarray:
    """Build (P,3) device array from genome indices; time via pulse_time(axis, delta_n)."""
    if not genome:
        return cp.zeros((0, 3), dtype=cp.float64)
    axes, dns, times = [], [], []
    for idx in genome:
        ax, dn = pool[int(idx)]
        axes.append(ax)
        dns.append(dn)
        times.append(pulse_time(ax, dn))  # CPU scalar ok; small overhead
    arr = np.column_stack([np.array(axes, float), np.array(dns, float), np.array(times, float)])
    return cp.asarray(arr, dtype=cp.float64)


def score_sequence(seq_gpu: cp.ndarray, cfg: GAConfig, res: cr.GPUResources, initial_mol: cp.ndarray) -> Tuple[float, int]:
    """Return (penalized_score, raw_survivors_in_region)."""
    mol = initial_mol.copy()
    cr.raman_cool_with_pumping(mol, seq_gpu, res, K_max=int(cfg.K_max))

    n_x, n_y, n_z, state, spin, is_lost = mol[:, 0], mol[:, 1], mol[:, 2], mol[:,3], mol[:,4], mol[:, 5]
    mask = (is_lost == 0) & (state == 1) & (spin == 0) & (n_x == 0) & (n_y == 0) & (n_z < 3)
    raw = int(cp.count_nonzero(mask).get())
    penalized = raw - cfg.length_penalty * int(seq_gpu.shape[0]) * int(mol.shape[0])
    return float(penalized), raw


# ==============================
# GA main
# ==============================
def run_ga(cfg: GAConfig, outdir: Path, res: cr.GPUResources) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Load pool and seed genome
    pool = load_allowed_pulses_from_cfg(cfg)
    pool_size = len(pool)
    if pool_size == 0:
        raise ValueError("allowed_pulses is empty.")

    seed_genome = load_seed_genome_from_file(Path(cfg.seed_file), pool)
    initial_mol = cp.asarray(np.load("mol_post_XYZ2.npy"))
    # DEAP setup
    try: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception: pass
    try: creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception: pass

    toolbox = base.Toolbox()
    rng = np.random.default_rng(cfg.py_seed)

    def clamp_len(g: List[int]) -> None:
        if len(g) < cfg.min_len:
            need = cfg.min_len - len(g)
            g.extend(rng.integers(0, pool_size, size=need).tolist())
        if len(g) > cfg.max_len:
            del g[cfg.max_len:]

    def init_ind() -> creator.Individual:
        g = list(seed_genome)
        clamp_len(g)
        return creator.Individual(g)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: List[int], initial_mol: cp.ndarray) -> tuple:
        clamp_len(ind)
        seq_gpu = build_sequence_from_indices(ind, pool)
        score, raw = score_sequence(seq_gpu, cfg, res, initial_mol)
        ind._raw_survivors = raw
        ind._num_pulses = int(seq_gpu.shape[0])
        return (score,)

    toolbox.register("evaluate", evaluate)

    # 1-point crossover on integer lists
    def one_point_cx(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        n = min(len(a), len(b))
        if n < 2:
            return a, b
        k = int(rng.integers(1, n))
        a[:], b[:] = a[:k] + b[k:], b[:k] + a[k:]
        return a, b

    toolbox.register("mate", one_point_cx)

    # Mutation constrained to allowed_pulses
    def mutate(ind: List[int]):
        changed = False
        # replace one gene
        if rng.random() < cfg.p_replace_gene and len(ind) > 0:
            i = int(rng.integers(0, len(ind)))
            ind[i] = int(rng.integers(0, pool_size))
            changed = True
        # add a gene
        if rng.random() < cfg.p_add_gene and len(ind) < cfg.max_len:
            pos = int(rng.integers(0, len(ind) + 1))
            ind.insert(pos, int(rng.integers(0, pool_size)))
            changed = True
        # drop a gene
        if rng.random() < cfg.p_drop_gene and len(ind) > cfg.min_len:
            pos = int(rng.integers(0, len(ind)))
            del ind[pos]
            changed = True
        if not changed and len(ind) > 0:
            # ensure at least one change
            i = int(rng.integers(0, len(ind)))
            ind[i] = int(rng.integers(0, pool_size))
        clamp_len(ind)
        return (ind,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Init population / HOF / logs
    pop = toolbox.population(n=cfg.pop_size)

    # Force first individual to exact seed (clamped)
    seed_ind = creator.Individual(list(seed_genome))
    clamp_len(seed_ind)
    pop[0] = seed_ind

    hof = tools.HallOfFame(5, similar=lambda a, b: (len(a) == len(b)) and (np.all(np.array(a) == np.array(b))))
    history = {"gen": [], "best": [], "avg": [], "std": [], "min": [], "max": []}

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"XYZ3_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GA loop
    for gen in range(cfg.n_gen):
        print(f"Running generation {gen}")
        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            with tqdm(total=len(invalid), desc=f"Evaluating Gen {gen:03d}", leave=False, dynamic_ncols=True) as pbar:
                for ind in invalid:
                    ind.fitness.values = toolbox.evaluate(ind, initial_mol)
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
            seq_gpu = build_sequence_from_indices(ind, pool)
            seq_host = cp.asnumpy(seq_gpu)
            np.save(gen_dir / f"top{rank}_sequence.npy", seq_host)
            meta = {
                "rank": rank,
                "fitness_penalized": ind.fitness.values[0],
                "raw_survivors": getattr(ind, "_raw_survivors", None),
                "num_pulses": getattr(ind, "_num_pulses", int(seq_gpu.shape[0])),
                "genome_indices": list(map(int, ind)),
                "genome_pairs": [list(pool[i]) for i in ind],
                "length_penalty": cfg.length_penalty,
            }
            with open(gen_dir / f"top{rank}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # Update HOF
        hof.update(pop)

        # Selection → Variation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover (1-point) — invalidate fitness unconditionally when applied
        for i in range(0, len(offspring) - 1, 2):
            if rng.random() < cfg.cx_prob:
                toolbox.mate(offspring[i], offspring[i + 1])  # in-place
                # IMPORTANT: mark both children for re-evaluation
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation — we already ensure at least one change inside mutate()
        for i in range(len(offspring)):
            if rng.random() < cfg.mut_prob:
                toolbox.mutate(offspring[i])  # in-place
                # IMPORTANT: mark child for re-evaluation
                del offspring[i].fitness.values

        pop[:] = offspring


    # Final eval & HOF
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind, initial_mol)
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
        seq_gpu = build_sequence_from_indices(ind, pool)
        seq_host = cp.asnumpy(seq_gpu)
        np.save(final_dir / f"top{rank}_sequence.npy", seq_host)
        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "fitness_penalized": ind.fitness.values[0],
                    "raw_survivors": getattr(ind, "_raw_survivors", None),
                    "num_pulses": int(seq_gpu.shape[0]),
                    "genome_indices": list(map(int, ind)),
                    "genome_pairs": [list(pool[i]) for i in ind],
                    "length_penalty": cfg.length_penalty,
                },
                f,
                indent=2,
            )
    print(f"\nSaved run to: {run_dir}")


# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    M_dev = cr.load_m_table_device()        # upload to GPU
    res   = cr.resources_from_config(M_dev)
    outdir = Path("ga_runs")
    cfg = GAConfig(
        n_gen=40, pop_size=100, cx_prob=0.7, mut_prob=0.6,
        min_len=20, max_len=120,
        p_add_gene=0.20, p_drop_gene=0.40, p_replace_gene=0.40,
        length_penalty=1/1000,
        n_molecules=50_000, temp=(25e-6, 25e-6, 25e-6), K_max=30,
        seed_file="XYZ3_original.npy",
        allowed_pulses = [
            (0,-4), (0,-3), (0,-2),(0,-1),
            (1,-4), (1,-3),(1,-2),(1,-1),
            (2,-9), (2,-8), (2,-7), (2,-6), (2,-5), (2,-4), (2,-3), (2,-2), (2,-1)],
        config_json=None,
        py_seed=123,
    )
    run_ga(cfg, outdir, res)
