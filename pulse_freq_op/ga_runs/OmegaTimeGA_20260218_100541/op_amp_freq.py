# ga_optimize_omega_time.py
from __future__ import annotations
import json
import time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import cupy as cp
from deap import base, creator, tools

import cu_rsc as cr
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
    omega_scale_sigma_init: float = 0.2    # initial sigma for Ω scale
    t_scale_sigma_init: float = 0.2        # initial sigma for t scale
    det_offset_sigma_init: float = 5000.0  # initial sigma for detuning offsets (Hz)

    # Bounds on scale factors
    omega_scale_min: float = 0.5
    omega_scale_max: float = 1.5
    t_scale_min: float = 0.5
    t_scale_max: float = 1.5
    pulse_freq_lim: float = 5e3

    # Penalty
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30

    # AOD trap parameters
    trap_shift: int = -2e3
    carrier_shift: int = 3e3
    trap_shift_sigma: int = 300
    carrier_shift_sigma: int = 300

    # File with base sequence (axis, delta_n, Omega_lin_Hz, t_sec, detuning_Hz)
    base_seq_npy: str = "base_seq.npy"

    outdir: str = "ga_runs"


# ==============================
# Helpers
# ==============================

def clamp_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def build_omega_time_det_mapping(base_seq: np.ndarray):
    """
    base_seq: (P,5) [axis, delta_n, Omega_lin_Hz, t_sec, detuning_Hz]

    Returns:
      unique_pairs: (M,2)
      inverse: (P,)
      baseline_omega_lin: (M,)
      baseline_t_sec:     (M,)
      baseline_det_lin:   (M,) detuning (Hz) per pulse type
    """
    if base_seq.ndim != 2 or base_seq.shape[1] < 5:
        raise ValueError("base_seq must be (P,5): [axis, delta_n, Omega_lin_Hz, t_sec, detuning_Hz]")

    axes      = base_seq[:, 0].astype(int)
    dns       = base_seq[:, 1].astype(int)
    omega_lin = base_seq[:, 2].astype(float)
    t_sec     = base_seq[:, 3].astype(float)
    det_lin   = base_seq[:, 4].astype(float)

    pairs = np.stack([axes, dns], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

    M = unique_pairs.shape[0]
    baseline_omega_lin = np.zeros(M, dtype=float)
    baseline_t_sec     = np.zeros(M, dtype=float)
    baseline_det_lin   = np.zeros(M, dtype=float)

    for k in range(M):
        mask = (inverse == k)
        baseline_omega_lin[k] = float(np.mean(omega_lin[mask]))
        baseline_t_sec[k]     = float(np.mean(t_sec[mask]))
        baseline_det_lin[k]   = float(np.mean(det_lin[mask]))

    return unique_pairs, inverse, baseline_omega_lin, baseline_t_sec, baseline_det_lin


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

    return score_count, good_fraction, nz_bar


def _inflate_genes_to_full_M(
    genes_z: np.ndarray,
    *,
    z_type_mask: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    genes_z: (3*Mz,) genes for Z-axis pulse *types* only.

    Returns full-length (M,) arrays:
      omega_scales_full, t_scales_full, det_offsets_full
    Non-Z entries are fixed at baseline (scales=1, offsets=0).
    """
    genes_z = np.asarray(genes_z, dtype=float)
    z_idx = np.nonzero(z_type_mask)[0]
    Mz = int(z_idx.size)
    if genes_z.shape[0] != 3 * Mz:
        raise ValueError(f"genes_z must have length 3*Mz={3*Mz}, got {genes_z.shape[0]}")

    omega_scales_full = np.ones(M, dtype=float)
    t_scales_full     = np.ones(M, dtype=float)
    det_offsets_full  = np.zeros(M, dtype=float)

    omega_scales_z = genes_z[:Mz]
    t_scales_z     = genes_z[Mz:2*Mz]
    det_offsets_z  = genes_z[2*Mz:3*Mz]

    omega_scales_full[z_idx] = omega_scales_z
    t_scales_full[z_idx]     = t_scales_z
    det_offsets_full[z_idx]  = det_offsets_z

    return omega_scales_full, t_scales_full, det_offsets_full


def score_sequence_from_scales(
    axes: np.ndarray,
    dns: np.ndarray,
    inverse: np.ndarray,
    baseline_omega_lin: np.ndarray,
    baseline_t_sec: np.ndarray,
    baseline_det_lin: np.ndarray,
    genes_z: np.ndarray,
    z_type_mask: np.ndarray,
    cfg: OmegaTimeGAConfig,
    res: cr.GPUResources,
):
    M = baseline_omega_lin.shape[0]

    omega_scales, t_scales, det_offsets = _inflate_genes_to_full_M(
        genes_z, z_type_mask=z_type_mask, M=M
    )

    omega_lin_vec = baseline_omega_lin * omega_scales
    t_sec_vec     = baseline_t_sec * t_scales
    det_lin_vec   = baseline_det_lin + det_offsets

    omega_lin_full = omega_lin_vec[inverse]
    t_sec_full     = t_sec_vec[inverse]
    det_full       = det_lin_vec[inverse]

    det_full = clamp_array(det_full, -float(cfg.pulse_freq_lim), float(cfg.pulse_freq_lim))

    pulses_host = np.column_stack([
        axes.astype(np.int32),
        dns.astype(np.int32),
        omega_lin_full.astype(np.float32),
        t_sec_full.astype(np.float32),
        det_full.astype(np.float32),
    ])

    mol_SLM = cr.build_thermal_molecules(
        int(cfg.n_molecules),
        list(cfg.temp),
        trap_detuning_mean=0.0,
        trap_detuning_sigma=float(cfg.trap_shift_sigma),
        carrier_detuning_mean=0.0,
        carrier_detuning_sigma=float(cfg.carrier_shift_sigma),
    )
    mol_AOD = cr.build_thermal_molecules(
        int(cfg.n_molecules),
        list(cfg.temp),
        trap_detuning_mean=float(cfg.trap_shift),
        trap_detuning_sigma=float(cfg.trap_shift_sigma),
        carrier_detuning_mean=float(cfg.carrier_shift),
        carrier_detuning_sigma=float(cfg.carrier_shift_sigma),
    )

    cr.raman_cool_with_pumping(
        molecules_dev=mol_SLM, pulses_dev=pulses_host, res=res,
        K_max=int(cfg.K_max), show_progress=False
    )
    cr.raman_cool_with_pumping(
        molecules_dev=mol_AOD, pulses_dev=pulses_host, res=res,
        K_max=int(cfg.K_max), show_progress=False
    )

    score_SLM, surv_SLM, nzbar_SLM = score_molecules(mol_SLM, max_nz=1)
    score_AOD, surv_AOD, nzbar_AOD = score_molecules(mol_AOD, max_nz=1)

    score_total = float(np.sqrt(score_SLM * score_AOD))

    return (score_total,
            score_SLM, surv_SLM, nzbar_SLM,
            score_AOD, surv_AOD, nzbar_AOD,
            omega_lin_vec, t_sec_vec, det_lin_vec,
            omega_scales, t_scales, det_offsets)


# ==============================
# GA main
# ==============================

def run_omega_time_ga(cfg: OmegaTimeGAConfig, res: cr.GPUResources) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_seq = np.load(cfg.base_seq_npy)
    if base_seq.shape[1] < 5:
        raise ValueError("Expected base_seq to be (P,5): [axis, delta_n, Omega_lin_Hz, t_sec, detuning_Hz]")

    axes = base_seq[:, 0].astype(int)
    dns  = base_seq[:, 1].astype(int)

    unique_pairs, inverse, baseline_omega_lin, baseline_t_sec, baseline_det_lin = build_omega_time_det_mapping(base_seq)
    M = unique_pairs.shape[0]

    # Only optimize Z-axis pulse TYPES (axis == 2)
    z_type_mask = (unique_pairs[:, 0] == 2)
    z_idx = np.nonzero(z_type_mask)[0]
    Mz = int(z_idx.size)
    print(f"Found {M} unique pulse types total; optimizing {Mz} Z-axis pulse types (axis==2).")

    rng = np.random.default_rng(cfg.py_seed)

    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)

    # Genome only for Z types: [omega_scales_z (Mz), t_scales_z (Mz), det_offsets_z (Mz)]
    def init_ind() -> creator.Individual:
        omega_scales_z = clamp_array(
            rng.normal(1.0, cfg.omega_scale_sigma_init, size=Mz),
            cfg.omega_scale_min, cfg.omega_scale_max
        )
        t_scales_z = clamp_array(
            rng.normal(1.0, cfg.t_scale_sigma_init, size=Mz),
            cfg.t_scale_min, cfg.t_scale_max
        )
        det_offsets_z = clamp_array(
            rng.normal(0.0, cfg.det_offset_sigma_init, size=Mz),
            -cfg.pulse_freq_lim, cfg.pulse_freq_lim
        )
        vec = np.concatenate([omega_scales_z, t_scales_z, det_offsets_z])
        return creator.Individual(vec.astype(float))

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: np.ndarray) -> tuple:
        (score_total,
         score_SLM, surv_SLM, nzbar_SLM,
         score_AOD, surv_AOD, nzbar_AOD,
         omega_vec, time_vec, det_vec,
         omega_scales_full, t_scales_full, det_offsets_full) = score_sequence_from_scales(
            axes, dns, inverse,
            baseline_omega_lin, baseline_t_sec, baseline_det_lin,
            np.asarray(ind, dtype=float),
            z_type_mask,
            cfg, res
        )

        ind.score_SLM = score_SLM
        ind.score_AOD = score_AOD
        ind.surv_SLM = surv_SLM
        ind.surv_AOD = surv_AOD
        ind.nzbar_SLM = nzbar_SLM
        ind.nzbar_AOD = nzbar_AOD

        # Keep output format the same: these are full-length M vectors
        ind.omega_vector = omega_vec.tolist()
        ind.time_vector  = time_vec.tolist()
        ind.detuning_vector = det_vec.tolist()
        ind.omega_scales_full = omega_scales_full
        ind.time_scales_full  = t_scales_full
        ind.det_offsets_full  = det_offsets_full
        return (score_total,)

    toolbox.register("evaluate", evaluate)

    def mate(a: np.ndarray, b: np.ndarray):
        alpha = rng.uniform(0.25, 0.75, size=a.shape[0])
        c1 = alpha * a + (1 - alpha) * b
        c2 = alpha * b + (1 - alpha) * a

        # Clamp within Z-genome bounds
        c1[:Mz]        = clamp_array(c1[:Mz],        cfg.omega_scale_min, cfg.omega_scale_max)
        c2[:Mz]        = clamp_array(c2[:Mz],        cfg.omega_scale_min, cfg.omega_scale_max)
        c1[Mz:2*Mz]    = clamp_array(c1[Mz:2*Mz],    cfg.t_scale_min,     cfg.t_scale_max)
        c2[Mz:2*Mz]    = clamp_array(c2[Mz:2*Mz],    cfg.t_scale_min,     cfg.t_scale_max)
        c1[2*Mz:3*Mz]  = clamp_array(c1[2*Mz:3*Mz],  -cfg.pulse_freq_lim, cfg.pulse_freq_lim)
        c2[2*Mz:3*Mz]  = clamp_array(c2[2*Mz:3*Mz],  -cfg.pulse_freq_lim, cfg.pulse_freq_lim)

        a[:] = c1
        b[:] = c2
        return a, b

    toolbox.register("mate", mate)

    def mutate(ind: np.ndarray):
        ind[:Mz] = clamp_array(
            ind[:Mz] + rng.normal(0.0, cfg.omega_scale_sigma_init, size=Mz),
            cfg.omega_scale_min, cfg.omega_scale_max
        )
        ind[Mz:2*Mz] = clamp_array(
            ind[Mz:2*Mz] + rng.normal(0.0, cfg.t_scale_sigma_init, size=Mz),
            cfg.t_scale_min, cfg.t_scale_max
        )
        ind[2*Mz:3*Mz] = clamp_array(
            ind[2*Mz:3*Mz] + rng.normal(0.0, cfg.det_offset_sigma_init, size=Mz),
            -cfg.pulse_freq_lim, cfg.pulse_freq_lim
        )
        return (ind,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=cfg.pop_size)

    # Baseline individual (Z only): scales=1, offsets=0
    baseline_ind = creator.Individual(
        np.concatenate([np.ones(Mz), np.ones(Mz), np.zeros(Mz)]).astype(float)
    )
    pop[0] = baseline_ind

    hof = tools.HallOfFame(5, similar=lambda a, b: np.allclose(a, b, rtol=1e-8, atol=1e-8))
    history = {"gen": [], "best": [], "avg": [], "std": [], "min": [], "max": []}

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"OmegaTimeGA_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for gen in range(cfg.n_gen):
        print(f"Running Omega+Time GA generation {gen}")

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

        gen_dir = run_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        best_ind = pop[int(np.argmax(fits))]
        np.save(gen_dir / "best_scales_vec.npy", np.asarray(best_ind))

        # Build & save corresponding full (P,5) sequence (format unchanged)
        genes_z = np.asarray(best_ind, dtype=float)
        omega_scales_full, t_scales_full, det_offsets_full = _inflate_genes_to_full_M(
            genes_z, z_type_mask=z_type_mask, M=M
        )

        omega_lin_vec = baseline_omega_lin * omega_scales_full
        t_sec_vec     = baseline_t_sec * t_scales_full
        det_lin_vec   = baseline_det_lin + det_offsets_full

        omega_lin_full = omega_lin_vec[inverse]
        t_sec_full     = t_sec_vec[inverse]
        det_full       = clamp_array(det_lin_vec[inverse], -float(cfg.pulse_freq_lim), float(cfg.pulse_freq_lim))

        best_seq_host = np.column_stack([
            axes.astype(np.int32),
            dns.astype(np.int32),
            omega_lin_full.astype(np.float32),
            t_sec_full.astype(np.float32),
            det_full.astype(np.float32),
        ])
        np.save(gen_dir / "best_sequence.npy", best_seq_host)

        meta = {
            "gen": gen,
            "score": float(best_ind.fitness.values[0]),
            "score_SLM": getattr(best_ind, "score_SLM", None),
            "score_AOD": getattr(best_ind, "score_AOD", None),
            "survive_SLM": getattr(best_ind, "surv_SLM", None),
            "survive_AOD": getattr(best_ind, "surv_AOD", None),
            "nzbar_SLM": getattr(best_ind, "nzbar_SLM", None),
            "nzbar_AOD": getattr(best_ind, "nzbar_AOD", None),
            "num_pulses": int(base_seq.shape[0]),
            "unique_pairs": unique_pairs.tolist(),
            # Keep same meta formats: full-length M vectors
            "omega_vector": omega_lin_vec.tolist(),
            "time_vector": t_sec_vec.tolist(),
            "detuning_vector": det_lin_vec.tolist(),
            "omega_scales": omega_scales_full.tolist(),
            "time_scales": t_scales_full.tolist(),
            "detuning_offsets": det_offsets_full.tolist(),
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

    np.save(run_dir / "history_gen.npy",  np.array(history["gen"]))
    np.save(run_dir / "history_best.npy", np.array(history["best"]))
    np.save(run_dir / "history_avg.npy",  np.array(history["avg"]))
    np.save(run_dir / "history_std.npy",  np.array(history["std"]))
    np.save(run_dir / "history_min.npy",  np.array(history["min"]))
    np.save(run_dir / "history_max.npy",  np.array(history["max"]))

    final_dir = run_dir / "final_top5"
    final_dir.mkdir(parents=True, exist_ok=True)
    for rank, ind in enumerate(hof, 1):
        genes_z = np.asarray(ind, dtype=float)
        omega_scales_full, t_scales_full, det_offsets_full = _inflate_genes_to_full_M(
            genes_z, z_type_mask=z_type_mask, M=M
        )

        omega_lin_vec = baseline_omega_lin * omega_scales_full
        t_sec_vec     = baseline_t_sec * t_scales_full
        det_lin_vec   = baseline_det_lin + det_offsets_full

        omega_lin_full = omega_lin_vec[inverse]
        t_sec_full     = t_sec_vec[inverse]
        det_full       = clamp_array(det_lin_vec[inverse], -float(cfg.pulse_freq_lim), float(cfg.pulse_freq_lim))

        full_seq = np.column_stack([
            axes.astype(np.int32),
            dns.astype(np.int32),
            omega_lin_full.astype(np.float32),
            t_sec_full.astype(np.float32),
            det_full.astype(np.float32),
        ])
        np.save(final_dir / f"top{rank}_sequence.npy", full_seq)

        meta = {
            "rank": rank,
            "score": float(ind.fitness.values[0]),
            "score_SLM": getattr(ind, "score_SLM", None),
            "score_AOD": getattr(ind, "score_AOD", None),
            "survive_SLM": getattr(ind, "surv_SLM", None),
            "survive_AOD": getattr(ind, "surv_AOD", None),
            "nzbar_SLM": getattr(ind, "nzbar_SLM", None),
            "nzbar_AOD": getattr(ind, "nzbar_AOD", None),
            "num_pulses": int(base_seq.shape[0]),
            "unique_pairs": unique_pairs.tolist(),
            "omega_vector": omega_lin_vec.tolist(),
            "time_vector": t_sec_vec.tolist(),
            "detuning_vector": det_lin_vec.tolist(),
            "omega_scales": omega_scales_full.tolist(),
            "time_scales": t_scales_full.tolist(),
            "detuning_offsets": det_offsets_full.tolist(),
        }
        with open(final_dir / f"top{rank}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nSaved Omega+Time GA run to: {run_dir}")


# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    M_dev = cr.load_m_table_device()
    res   = cr.resources_from_config(M_dev)

    cfg = OmegaTimeGAConfig(
        n_gen=50,
        pop_size=128,
        cx_prob=0.7,
        mut_prob=0.8,
        py_seed=42,
        n_molecules=10_000,
        temp=(25e-6, 25e-6, 25e-6),
        K_max=30,
        base_seq_npy="base_seq.npy",
        outdir="ga_runs",
        omega_scale_min=0.001,
        omega_scale_max=10.0,
        t_scale_min=0.001,
        t_scale_max=10.0,
        pulse_freq_lim=5e3,
        trap_shift=-2e3,
        carrier_shift=3e3,
        trap_shift_sigma=300,
        carrier_shift_sigma=300
    )
    run_omega_time_ga(cfg, res)
