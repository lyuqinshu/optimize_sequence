# ga_optimize_omega.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cupy as cp
from deap import base, creator, tools

import cu_rsc as cr
cr.setup_tables()


# ==============================
# Config
# ==============================
@dataclass
class AmpGAConfig:
    # GA params
    n_gen: int = 40
    pop_size: int = 64
    cx_prob: float = 0.7
    mut_prob: float = 0.8
    py_seed: int = 123

    # Mutation (Gaussian)
    sigma_init: float = 0.5       # initial mutation scale (in units of Omega*t)
    sigma_min: float = 0.05
    sigma_max: float = 2.0

    # Bounds on Omega*t
    omega_min: float = 0.0
    omega_max: float = 15.0

    # Penalty (same form as before, if you want it)
    length_penalty: float = 0.001
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    # File with base sequence (axis, delta_n, Omega*t)
    base_seq_npy: str = "best_seq.npy"

    outdir: str = "amp_ga_runs"


# ==============================
# Helpers
# ==============================
def build_omega_mapping(base_seq: np.ndarray):
    """
    base_seq: (P,3) array: [axis, delta_n, Omega*t]
    Returns:
      unique_pairs: (M,2) array of unique (axis, delta_n)
      inverse: (P,) array, mapping pulse index -> pair index
      baseline_omega: (M,) baseline Omega*t per unique pair
    """
    if base_seq.ndim != 2 or base_seq.shape[1] < 3:
        raise ValueError("base_seq must be (P,3)")

    axes = base_seq[:, 0].astype(int)
    dns  = base_seq[:, 1].astype(int)
    omega = base_seq[:, 2].astype(float)

    pairs = np.stack([axes, dns], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

    M = unique_pairs.shape[0]
    baseline_omega = np.zeros(M, dtype=float)
    for idx_pair in range(M):
        mask = (inverse == idx_pair)
        baseline_omega[idx_pair] = float(np.mean(omega[mask]))

    return unique_pairs, inverse, baseline_omega


def clamp_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def score_sequence_from_omega(
    axes: np.ndarray,
    dns: np.ndarray,
    inverse: np.ndarray,
    omega_vec: np.ndarray,
    cfg: AmpGAConfig,
    res: cr.GPUResources,
) -> Tuple[float, int]:
    """
    axes, dns: (P,) arrays (int or float)
    inverse: (P,) mapping pulses -> pair index
    omega_vec: (M,) array, Omega*t per unique pair

    Returns (penalized_score, raw_survivors)
    """
    # Build full sequence
    omega_full = omega_vec[inverse]  # broadcast
    host = np.column_stack(
        [axes.astype(float), dns.astype(float), omega_full.astype(float)]
    )
    seq_gpu = cp.asarray(host, dtype=cp.float64)

    # Run cooling simulation
    mol = cr.build_thermal_molecules_gpu(int(cfg.n_molecules), list(cfg.temp))
    cr.raman_cool_with_pumping(mol, seq_gpu, res, K_max=int(cfg.K_max), show_progress=True)

    n_x, n_y, n_z = mol[:, 0], mol[:, 1], mol[:, 2]
    is_lost, spin, mN = mol[:, 5], mol[:, 4], mol[:, 3]

    mask = (is_lost == 0) & (n_x == 0) & (n_y == 0) & (n_z == 0) & (spin == 0) & (mN == 1)
    raw = int(cp.count_nonzero(mask).get())

    # Length penalty like before
    total_pulses = int(host.shape[0])
    penalty = total_pulses * cfg.length_penalty * int(cfg.n_molecules)
    penalized = float(raw) - float(penalty)
    return penalized, raw


# ==============================
# GA main
# ==============================
def run_amp_ga(cfg: AmpGAConfig, res: cr.GPUResources) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load base sequence
    base_seq = np.load(cfg.base_seq_npy)
    axes = base_seq[:, 0].astype(int)
    dns  = base_seq[:, 1].astype(int)

    unique_pairs, inverse, baseline_omega = build_omega_mapping(base_seq)
    M = unique_pairs.shape[0]

    print(f"Found {M} unique (axis, delta_n) pulse types in base sequence.")

    rng = np.random.default_rng(cfg.py_seed)

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

    # Initialize individual near baseline with Gaussian noise
    def init_ind() -> creator.Individual:
        noise = rng.normal(loc=0.0, scale=cfg.sigma_init, size=M)
        vec = baseline_omega + noise
        vec = clamp_array(vec, cfg.omega_min, cfg.omega_max)
        return creator.Individual(vec.astype(float))

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: np.ndarray) -> tuple:
        omega_vec = np.asarray(ind, dtype=float)
        score, raw = score_sequence_from_omega(
            axes, dns, inverse, omega_vec, cfg, res
        )
        # stash extra info if you want to save later
        ind._raw_survivors = raw
        ind._num_pulses = int(base_seq.shape[0])
        return (score,)

    toolbox.register("evaluate", evaluate)

    # Crossover: blend arithmetic crossover
    def mate(a: np.ndarray, b: np.ndarray):
        alpha = rng.uniform(0.25, 0.75, size=a.shape[0])
        child1 = alpha * a + (1 - alpha) * b
        child2 = alpha * b + (1 - alpha) * a
        child1 = clamp_array(child1, cfg.omega_min, cfg.omega_max)
        child2 = clamp_array(child2, cfg.omega_min, cfg.omega_max)
        a[:] = child1
        b[:] = child2
        return a, b

    toolbox.register("mate", mate)

    # Mutation: Gaussian noise per gene, scale adapting to distance from baseline
    def mutate(ind: np.ndarray):
        # Example: sigma scaled by |ind - baseline|, clamped
        diff = np.abs(ind - baseline_omega)
        sigma = np.clip(cfg.sigma_init * (1.0 + diff / (np.mean(diff) + 1e-6)),
                        cfg.sigma_min, cfg.sigma_max)

        noise = rng.normal(loc=0.0, scale=sigma, size=ind.shape[0])
        ind[:] = clamp_array(ind + noise, cfg.omega_min, cfg.omega_max)
        return (ind,)

    toolbox.register("mutate", mutate)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # Init population / HOF / logs
    pop = toolbox.population(n=cfg.pop_size)

    # Force one individual to exactly baseline (for reference)
    baseline_ind = creator.Individual(baseline_omega.copy())
    pop[0] = baseline_ind

    hof = tools.HallOfFame(5, similar=lambda a, b: np.allclose(a, b, rtol=1e-8, atol=1e-8))
    history = {"gen": [], "best": [], "avg": [], "std": [], "min": [], "max": []}

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"AMPGA_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GA loop
    for gen in range(cfg.n_gen):
        print(f"Running amplitude GA generation {gen}")

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best, avg = float(fits.max()), float(fits.mean())
        std = float(fits.std(ddof=1)) if fits.size > 1 else 0.0

        history["gen"].append(gen)
        history["best"].append(best)
        history["avg"].append(avg)
        history["std"].append(std)
        history["min"].append(float(fits.min()))
        history["max"].append(float(fits.max()))

        print(f"[Gen {gen:03d}] best={best:.1f} avg={avg:.1f} std={std:.1f}")

        # Save best individual of this gen (for debugging)
        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        best_ind = pop[int(np.argmax(fits))]
        np.save(gen_dir / "best_omega_vec.npy", np.asarray(best_ind))

        # Build & save the corresponding full sequence
        best_seq_host = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            np.asarray(best_ind)[inverse].astype(float),
        ])
        np.save(gen_dir / "best_sequence.npy", best_seq_host)

        meta = {
            "gen": gen,
            "fitness_penalized": float(best_ind.fitness.values[0]),
            "raw_survivors": getattr(best_ind, "_raw_survivors", None),
            "num_pulses": int(base_seq.shape[0]),
            "length_penalty": cfg.length_penalty,
            "unique_pairs": unique_pairs.tolist(),
        }
        with open(gen_dir / "meta.json", "w") as f:
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

    # Save final HoF
    final_dir = run_dir / "final_top5"
    final_dir.mkdir(parents=True, exist_ok=True)
    for rank, ind in enumerate(hof, 1):
        omega_vec = np.asarray(ind)
        full_seq = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            omega_vec[inverse].astype(float),
        ])
        np.save(final_dir / f"top{rank}_sequence.npy", full_seq)
        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "fitness_penalized": float(ind.fitness.values[0]),
                    "raw_survivors": getattr(ind, "_raw_survivors", None),
                    "num_pulses": int(base_seq.shape[0]),
                    "length_penalty": cfg.length_penalty,
                    "unique_pairs": unique_pairs.tolist(),
                },
                f,
                indent=2,
            )

    print(f"\nSaved amplitude GA run to: {run_dir}")


# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    # GPU resources as in your previous script
    M_dev = cr.load_m_table_device()
    res   = cr.resources_from_config(M_dev)

    cfg = AmpGAConfig(
        n_gen=40,
        pop_size=64,
        cx_prob=0.7,
        mut_prob=0.8,
        py_seed=42,
        sigma_init=0.5,
        sigma_min=0.05,
        sigma_max=2.0,
        omega_min=0.0,
        omega_max=30.0,
        length_penalty=0.0,
        n_molecules=50_000,
        temp=(25e-6, 25e-6, 25e-6),
        K_max=30,
        base_seq_npy="sequence_optimized.npy",
        outdir="amp_ga_runs",
    )
    run_amp_ga(cfg, res)
