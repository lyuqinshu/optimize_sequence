# ga_optimize_omega_time.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import cupy as cp
from deap import base, creator, tools
from dataclasses import field
import cu_rsc as cr
from tqdm import tqdm
cr.setup_tables()

# ==============================
# Config
# ==============================

@dataclass
class OmegaTimeGAConfig:
    # GA params
    n_gen: int = 40
    pop_size: int = 64
    cx_prob: float = 0.7
    mut_prob: float = 0.8
    py_seed: int = 123

    # Mutation (on scale factors, dimensionless)
    omega_scale_sigma_init: float = 0.2   # initial sigma for Ω scale
    t_scale_sigma_init: float = 0.2       # initial sigma for t scale

    # Bounds on scale factors
    omega_scale_min: float = 0.5
    omega_scale_max: float = 1.5
    t_scale_min: float = 0.5
    t_scale_max: float = 1.5

    # Penalty
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    # File with base sequence (axis, delta_n, Omega_lin_Hz, t_sec)
    base_seq_npy: str = "base_seq_omega_time.npy"

    outdir: str = "omega_time_ga_runs"


# ==============================
# Helpers
# ==============================

def clamp_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def build_omega_time_mapping(base_seq: np.ndarray):
    """
    base_seq: (P,4) array: [axis, delta_n, Omega_lin_Hz, t_sec]

    Returns:
      unique_pairs: (M,2) array of unique (axis, delta_n)
      inverse: (P,) array, mapping pulse index -> pair index
      baseline_omega_lin: (M,) baseline Ω_lin (Hz) per unique pair
      baseline_t_sec:     (M,) baseline t_sec (s) per unique pair
    """
    if base_seq.ndim != 2 or base_seq.shape[1] < 4:
        raise ValueError("base_seq must be (P,4): [axis, delta_n, Omega_lin_Hz, t_sec]")

    axes       = base_seq[:, 0].astype(int)
    dns        = base_seq[:, 1].astype(int)
    omega_lin  = base_seq[:, 2].astype(float)  # Hz
    t_sec_full = base_seq[:, 3].astype(float)  # s

    pairs = np.stack([axes, dns], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

    M = unique_pairs.shape[0]
    baseline_omega_lin = np.zeros(M, dtype=float)
    baseline_t_sec     = np.zeros(M, dtype=float)

    for idx_pair in range(M):
        mask = (inverse == idx_pair)
        if not np.any(mask):
            raise RuntimeError("Mapping logic error: no pulses for pair index")

        # Average over all pulses of this type (in case there is slight variation)
        baseline_omega_lin[idx_pair] = float(np.mean(omega_lin[mask]))
        baseline_t_sec[idx_pair]     = float(np.mean(t_sec_full[mask]))

    return unique_pairs, inverse, baseline_omega_lin, baseline_t_sec

def build_z_omega_time_mapping(base_seq: np.ndarray):
    """
    Like build_omega_time_mapping, but only for Z pulses (axis==2).

    Returns:
      z_dns: (Mz,) unique delta_n values for axis==2
      z_inverse: (Pz,) mapping of Z-pulse indices -> z_dns index
      z_idx: (Pz,) the indices into the full pulse list that are Z pulses
      z_baseline_omega_lin: (Mz,) baseline Ω for each z delta_n
      z_baseline_t_sec:     (Mz,) baseline t for each z delta_n
      baseline_omega_full:  (P,) per-pulse baseline Ω from base_seq
      baseline_t_full:      (P,) per-pulse baseline t from base_seq
    """
    if base_seq.ndim != 2 or base_seq.shape[1] < 4:
        raise ValueError("base_seq must be (P,4): [axis, delta_n, Omega_lin_Hz, t_sec]")

    axes = base_seq[:, 0].astype(int)
    dns  = base_seq[:, 1].astype(int)

    baseline_omega_full = base_seq[:, 2].astype(float)
    baseline_t_full     = base_seq[:, 3].astype(float)

    z_idx = np.where(axes == 2)[0]
    if z_idx.size == 0:
        raise ValueError("No Z pulses found (axis==2).")

    z_dns_all = dns[z_idx]
    z_dns, z_inverse = np.unique(z_dns_all, return_inverse=True)

    Mz = z_dns.shape[0]
    z_baseline_omega_lin = np.zeros(Mz, dtype=float)
    z_baseline_t_sec     = np.zeros(Mz, dtype=float)

    for j in range(Mz):
        mask = (z_inverse == j)
        # Average over all Z pulses with this delta_n
        z_baseline_omega_lin[j] = float(np.mean(baseline_omega_full[z_idx][mask]))
        z_baseline_t_sec[j]     = float(np.mean(baseline_t_full[z_idx][mask]))

    return (
        z_dns,
        z_inverse,
        z_idx,
        z_baseline_omega_lin,
        z_baseline_t_sec,
        baseline_omega_full,
        baseline_t_full,
    )


def score_molecules(
    mol: cp.ndarray,
    *,
    nxny_max: int = 3,
    w_bad: float = 100.0,     # penalty for not meeting "good" fraction
    w_lost: float = 50.0,    # extra penalty for losses
    empty_score: float = 1e6 # if no good molecules
) -> float:
    """
    Minimization score:
      - encourage survival (is_lost==0) and correct manifold (spin==0, mN==1)
      - encourage nx, ny <= nxny_max
      - minimize mean nz among the "good" molecules

    Returns a Python float (smaller is better).
    """
    n_x, n_y, n_z = mol[:, 0], mol[:, 1], mol[:, 2]
    is_lost, spin, mN = mol[:, 5], mol[:, 4], mol[:, 3]

    N = mol.shape[0]

    base_ok = (spin == 0) & (mN == 1)
    alive   = (is_lost == 0)

    good = alive & base_ok & (n_x <= nxny_max) & (n_y <= nxny_max)

    good_count = cp.count_nonzero(good).astype(cp.float64)
    lost_frac  = cp.count_nonzero(~alive).astype(cp.float64) / float(N)
    good_frac  = good_count / float(N)

    # mean nz on good subset
    avg_nz_good = cp.where(
        good_count > 0,
        cp.sum(n_z[good]).astype(cp.float64) / good_count,
        cp.asarray(float(empty_score), dtype=cp.float64),
    )

    score = avg_nz_good + w_bad * (1.0 - good_frac) + w_lost * lost_frac
    return float(score.get()), float(avg_nz_good.get()), float(good_frac.get()), float(1 - lost_frac.get())



def score_sequence_from_scales(
    axes: np.ndarray,
    dns: np.ndarray,
    z_inverse: np.ndarray,               # (Pz,)
    z_idx: np.ndarray,                   # (Pz,)
    z_baseline_omega_lin: np.ndarray,    # (Mz,)
    z_baseline_t_sec: np.ndarray,        # (Mz,)
    baseline_omega_full: np.ndarray,     # (P,)
    baseline_t_full: np.ndarray,         # (P,)
    scales_vec: np.ndarray,
    cfg: OmegaTimeGAConfig,
    res: cr.GPUResources,
):
    """
    scales_vec: (2*Mz,) = [omega_scales_z (Mz), t_scales_z (Mz)]
    Only affects pulses with axis==2. X/Y stay baseline.
    """
    Mz = z_baseline_omega_lin.shape[0]
    if scales_vec.shape[0] != 2 * Mz:
        raise ValueError("scales_vec must have length 2*Mz (Z-only)")

    scales_vec = np.asarray(scales_vec, dtype=float)
    omega_scales_z = scales_vec[:Mz]
    t_scales_z     = scales_vec[Mz:]

    # Start from per-pulse baseline (X/Y unchanged)
    omega_lin_full = baseline_omega_full.copy()
    t_sec_full     = baseline_t_full.copy()

    # For Z pulses: map each Z pulse -> its z_dns index and apply its scale
    omega_lin_full[z_idx] = z_baseline_omega_lin[z_inverse] * omega_scales_z[z_inverse]
    t_sec_full[z_idx]     = z_baseline_t_sec[z_inverse] * t_scales_z[z_inverse]

    pulses_host = np.column_stack([
        axes.astype(float),
        dns.astype(float),
        omega_lin_full.astype(float),
        t_sec_full.astype(float),
    ])

    mol = cr.build_thermal_molecules(int(cfg.n_molecules), list(cfg.temp))
    cr.raman_cool_with_pumping(
        molecules_dev=mol,
        pulses_dev=pulses_host,
        res=res,
        K_max=int(cfg.K_max),
        show_progress=True,
    )
    return score_molecules(mol)



# ==============================
# GA main
# ==============================

def run_omega_time_ga(cfg: OmegaTimeGAConfig, res: cr.GPUResources) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load base sequence (axis, delta_n, Omega_lin_Hz, t_sec)
    base_seq = np.load(cfg.base_seq_npy)
    if base_seq.shape[1] < 4:
        raise ValueError("Expected base_seq to be (P,4): [axis, delta_n, Omega_lin_Hz, t_sec]")

    axes = base_seq[:, 0].astype(int)
    dns  = base_seq[:, 1].astype(int)

    (
    z_dns,
    z_inverse,
    z_idx,
    z_baseline_omega_lin,
    z_baseline_t_sec,
    baseline_omega_full,
    baseline_t_full,
    ) = build_z_omega_time_mapping(base_seq)

    Mz = z_dns.shape[0]
    print(f"Found {Mz} unique Z (delta_n) pulse types in base sequence (axis==2).")


    rng = np.random.default_rng(cfg.py_seed)

    # DEAP setup
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()

    # Initialize individual: scales around 1.0
    def init_ind() -> creator.Individual:
        omega_scales = rng.normal(loc=1.0, scale=cfg.omega_scale_sigma_init, size=Mz)
        omega_scales = clamp_array(omega_scales, cfg.omega_scale_min, cfg.omega_scale_max)

        t_scales = rng.normal(loc=1.0, scale=cfg.t_scale_sigma_init, size=Mz)
        t_scales = clamp_array(t_scales, cfg.t_scale_min, cfg.t_scale_max)

        vec = np.concatenate([omega_scales, t_scales])
        return creator.Individual(vec.astype(float))


    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: np.ndarray) -> tuple:
        scales_vec = np.asarray(ind, dtype=float)
        scores, z_mean, good_frac, survive_frac = score_sequence_from_scales(
            axes,
            dns,
            z_inverse,
            z_idx,
            z_baseline_omega_lin,
            z_baseline_t_sec,
            baseline_omega_full,
            baseline_t_full,
            scales_vec,
            cfg,
            res,
        )

        ind._num_pulses = int(base_seq.shape[0])
        ind.z_mean = z_mean
        ind.good_frac = good_frac
        ind.survive_frac = survive_frac
        return (scores,)

    toolbox.register("evaluate", evaluate)

    # Crossover: arithmetic on scale vectors
    def mate(a: np.ndarray, b: np.ndarray):
        alpha = rng.uniform(0.25, 0.75, size=a.shape[0])
        child1 = alpha * a + (1 - alpha) * b
        child2 = alpha * b + (1 - alpha) * a

        M_local = Mz  # Z-only

        child1[:M_local] = clamp_array(child1[:M_local], cfg.omega_scale_min, cfg.omega_scale_max)
        child2[:M_local] = clamp_array(child2[:M_local], cfg.omega_scale_min, cfg.omega_scale_max)

        child1[M_local:] = clamp_array(child1[M_local:], cfg.t_scale_min, cfg.t_scale_max)
        child2[M_local:] = clamp_array(child2[M_local:], cfg.t_scale_min, cfg.t_scale_max)

        a[:] = child1
        b[:] = child2
        return a, b


    toolbox.register("mate", mate)

    # Mutation: add Gaussian noise to scale factors
    def mutate(ind: np.ndarray):
        M_local = Mz  # Z-only
        omega_scales = ind[:M_local]
        t_scales     = ind[M_local:]

        omega_scales = omega_scales + rng.normal(loc=0.0, scale=cfg.omega_scale_sigma_init, size=M_local)
        t_scales     = t_scales + rng.normal(loc=0.0, scale=cfg.t_scale_sigma_init, size=M_local)

        omega_scales = clamp_array(omega_scales, cfg.omega_scale_min, cfg.omega_scale_max)
        t_scales     = clamp_array(t_scales, cfg.t_scale_min, cfg.t_scale_max)

        ind[:M_local] = omega_scales
        ind[M_local:] = t_scales
        return (ind,)


    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Init population / HOF / logs
    pop = toolbox.population(n=cfg.pop_size)

    # Baseline individual: all scales = 1
    baseline_scales = np.concatenate([
        np.ones_like(z_baseline_omega_lin),
        np.ones_like(z_baseline_t_sec),
    ])

    baseline_ind = creator.Individual(baseline_scales.astype(float))
    pop[0] = baseline_ind

    hof = tools.HallOfFame(5, similar=lambda a, b: np.allclose(a, b, rtol=1e-8, atol=1e-8))
    history = {"gen": [], "best": [], "avg": [], "std": [], "min": [], "max": []}

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"OmegaTimeGA_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GA loop
    for gen in range(cfg.n_gen):
        print(f"Running Omega+Time GA generation {gen}")

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best, avg = float(fits.min()), float(fits.mean())
        std = float(fits.std(ddof=1)) if fits.size > 1 else 0.0

        history["gen"].append(gen)
        history["best"].append(best)
        history["avg"].append(avg)
        history["std"].append(std)
        history["min"].append(float(fits.min()))
        history["max"].append(float(fits.max()))

        print(f"[Gen {gen:03d}] best={best:.1f} avg={avg:.1f} std={std:.1f}")

        # Save best individual of this gen
        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        best_ind = pop[int(np.argmin(fits))]
        np.save(gen_dir / "best_scales_vec.npy", np.asarray(best_ind))

        Mz_local = z_baseline_omega_lin.shape[0]
        omega_scales_z = np.asarray(best_ind[:Mz_local])
        t_scales_z     = np.asarray(best_ind[Mz_local:])

        omega_lin_full = baseline_omega_full.copy()
        t_sec_full     = baseline_t_full.copy()

        omega_lin_full[z_idx] = z_baseline_omega_lin[z_inverse] * omega_scales_z[z_inverse]
        t_sec_full[z_idx]     = z_baseline_t_sec[z_inverse] * t_scales_z[z_inverse]

        best_seq_host = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            omega_lin_full.astype(float),
            t_sec_full.astype(float),
        ])
        np.save(gen_dir / "best_sequence.npy", best_seq_host)

        meta = {
            "gen": gen,
            "fitness": float(best_ind.fitness.values[0]),
            "z_mean": float(best_ind.z_mean),
            "good_frac": float(best_ind.good_frac),
            "survive_frac": float(best_ind.survive_frac),
            "num_pulses": int(base_seq.shape[0]),
            "z_dns": z_dns.tolist(),
            "omega_scales_z": omega_scales_z.tolist(),
            "time_scales_z": t_scales_z.tolist(),
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

    # Save final HoF sequences
    final_dir = run_dir / "final_top5"
    final_dir.mkdir(parents=True, exist_ok=True)

    for rank, ind in enumerate(hof, 1):
        scales = np.asarray(ind, dtype=float)
        Mz_local = Mz
        omega_scales_z = scales[:Mz_local]
        t_scales_z     = scales[Mz_local:]

        omega_lin_full = baseline_omega_full.copy()
        t_sec_full     = baseline_t_full.copy()

        omega_lin_full[z_idx] = z_baseline_omega_lin[z_inverse] * omega_scales_z[z_inverse]
        t_sec_full[z_idx]     = z_baseline_t_sec[z_inverse] * t_scales_z[z_inverse]

        full_seq = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            omega_lin_full.astype(float),
            t_sec_full.astype(float),
        ])
        np.save(final_dir / f"top{rank}_sequence.npy", full_seq)

        meta = {
            "rank": rank,
            "fitness": float(ind.fitness.values[0]),
            "z_mean": float(ind.z_mean),
            "good_frac": float(ind.good_frac),
            "survive_frac": float(ind.survive_frac),
            "num_pulses": int(base_seq.shape[0]),
            "z_dns": z_dns.tolist(),
            "omega_scales_z": omega_scales_z.tolist(),
            "time_scales_z": t_scales_z.tolist(),
        }
        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)




    print(f"\nSaved Omega+Time GA run to: {run_dir}")


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    # GPU resources as before
    M_dev = cr.load_m_table_device()
    res   = cr.resources_from_config(M_dev)

    cfg = OmegaTimeGAConfig(
        n_gen=40,
        pop_size=64,
        cx_prob=0.7,
        mut_prob=0.8,
        py_seed=42,
        n_molecules=10_000,
        temp=(25e-6, 25e-6, 25e-6),
        K_max=30,
        base_seq_npy="seq_partial.npy",
        outdir="omega_time_ga_runs",
        omega_scale_min=0.1,
        omega_scale_max=10.0,
        t_scale_min=0.1,
        t_scale_max=5.0,
    )
    run_omega_time_ga(cfg, res)
