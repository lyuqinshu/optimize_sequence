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

    trap_dets: Sequence[float] = field(default_factory=lambda: (-2000.0, -1000.0, 0.0, 1000.0, 2000.0))

    # Penalty
    n_molecules: int = 50_000
    temp: Tuple[float, float, float] = (25e-6, 25e-6, 25e-6)
    K_max: int = 30
    n_good: int = 3
    std_penalty: float = 0.5

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

def score_molecules(
    mol: cp.ndarray,
    cfg: OmegaTimeGAConfig,
):
    n_good = cfg.n_good
    n_x, n_y, n_z = mol[:, 0], mol[:, 1], mol[:, 2]
    is_lost, spin, mN = mol[:, 5], mol[:, 4], mol[:, 3]
    mask = (is_lost == 0) & (n_x <= n_good) & (n_y <= n_good) & (n_z <= n_good) & (spin == 0) & (mN == 1)
    raw = int(cp.count_nonzero(mask).get())

    return raw

def score_sequence_from_scales(
    axes: np.ndarray,
    dns: np.ndarray,
    inverse: np.ndarray,
    baseline_omega_lin: np.ndarray,
    baseline_t_sec: np.ndarray,
    scales_vec: np.ndarray,
    cfg: OmegaTimeGAConfig,
    res: cr.GPUResources,
):
    """
    axes, dns: (P,) arrays (int)
    inverse: (P,) mapping pulses -> pair index
    baseline_omega_lin: (M,) baseline Ω_lin (Hz)
    baseline_t_sec:     (M,) baseline t_sec (s)
    scales_vec: (2M,) array: [omega_scales (M), t_scales (M)]

    Returns (good_list, scroe)
    """
    M = baseline_omega_lin.shape[0]
    if scales_vec.shape[0] != 2 * M:
        raise ValueError("scales_vec must have length 2*M")

    scales_vec = np.asarray(scales_vec, dtype=float)

    omega_scales = scales_vec[:M]
    t_scales     = scales_vec[M:]

    # Actual Ω_lin and t_sec per pair
    omega_lin_vec = baseline_omega_lin * omega_scales
    t_sec_vec     = baseline_t_sec * t_scales

    # Broadcast to full sequence via inverse map
    omega_lin_full = omega_lin_vec[inverse]
    t_sec_full     = t_sec_vec[inverse]

    # Build full (P,4) sequence on host: [axis, delta_n, Omega_lin_Hz, t_sec]
    pulses_host = np.column_stack([
        axes.astype(float),
        dns.astype(float),
        omega_lin_full.astype(float),
        t_sec_full.astype(float),
    ])

    # Run cooling simulation

    goods = []
    for det in cfg.trap_dets:
        mol = cr.build_thermal_molecules_gpu(int(cfg.n_molecules), list(cfg.temp))
        cr.raman_cool_with_pumping(
            molecules_dev=mol,
            pulses_dev=pulses_host,
            res=res,
            K_max=int(cfg.K_max),
            show_progress=True,
            trap_detuning=(0, 0, det),
        )
        good = score_molecules(
            mol, cfg
        )
        goods.append(good)

    score = np.mean(goods) - cfg.std_penalty * np.std(goods)
    
    return goods, score


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

    # Build mapping: per unique (axis, delta_n) we get baseline Ω_lin and t_sec
    unique_pairs, inverse, baseline_omega_lin, baseline_t_sec = build_omega_time_mapping(
        base_seq
        # duration_matrix no longer needed, baseline comes directly from base_seq
    )
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

    # Initialize individual: scales around 1.0
    def init_ind() -> creator.Individual:
        omega_scales = rng.normal(loc=1.0, scale=cfg.omega_scale_sigma_init, size=M)
        omega_scales = clamp_array(omega_scales, cfg.omega_scale_min, cfg.omega_scale_max)

        t_scales = rng.normal(loc=1.0, scale=cfg.t_scale_sigma_init, size=M)
        t_scales = clamp_array(t_scales, cfg.t_scale_min, cfg.t_scale_max)

        vec = np.concatenate([omega_scales, t_scales])
        return creator.Individual(vec.astype(float))

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind: np.ndarray) -> tuple:
        scales_vec = np.asarray(ind, dtype=float)
        goods, scores = score_sequence_from_scales(
            axes,
            dns,
            inverse,
            baseline_omega_lin,
            baseline_t_sec,
            scales_vec,
            cfg,
            res,
        )
        ind.goods = goods
        ind._num_pulses = int(base_seq.shape[0])
        return (scores,)

    toolbox.register("evaluate", evaluate)

    # Crossover: arithmetic on scale vectors
    def mate(a: np.ndarray, b: np.ndarray):
        alpha = rng.uniform(0.25, 0.75, size=a.shape[0])
        child1 = alpha * a + (1 - alpha) * b
        child2 = alpha * b + (1 - alpha) * a

        M_local = baseline_omega_lin.shape[0]

        # Clamp Ω scales
        child1[:M_local] = clamp_array(child1[:M_local], cfg.omega_scale_min, cfg.omega_scale_max)
        child2[:M_local] = clamp_array(child2[:M_local], cfg.omega_scale_min, cfg.omega_scale_max)
        # Clamp t scales
        child1[M_local:] = clamp_array(child1[M_local:], cfg.t_scale_min, cfg.t_scale_max)
        child2[M_local:] = clamp_array(child2[M_local:], cfg.t_scale_min, cfg.t_scale_max)

        a[:] = child1
        b[:] = child2
        return a, b

    toolbox.register("mate", mate)

    # Mutation: add Gaussian noise to scale factors
    def mutate(ind: np.ndarray):
        M_local = baseline_omega_lin.shape[0]
        omega_scales = ind[:M_local]
        t_scales     = ind[M_local:]

        omega_scales = omega_scales + rng.normal(
            loc=0.0, scale=cfg.omega_scale_sigma_init, size=M_local
        )
        t_scales = t_scales + rng.normal(
            loc=0.0, scale=cfg.t_scale_sigma_init, size=M_local
        )

        omega_scales = clamp_array(omega_scales, cfg.omega_scale_min, cfg.omega_scale_max)
        t_scales = clamp_array(t_scales, cfg.t_scale_min, cfg.t_scale_max)

        ind[:M_local] = omega_scales
        ind[M_local:] = t_scales
        return (ind,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Init population / HOF / logs
    pop = toolbox.population(n=cfg.pop_size)

    # Baseline individual: all scales = 1
    baseline_scales = np.concatenate([
        np.ones_like(baseline_omega_lin),
        np.ones_like(baseline_t_sec),
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
        best, avg = float(fits.max()), float(fits.mean())
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
        best_ind = pop[int(np.argmax(fits))]
        np.save(gen_dir / "best_scales_vec.npy", np.asarray(best_ind))

        # Build & save corresponding full (P,4) sequence
        M_local = baseline_omega_lin.shape[0]
        omega_scales = np.asarray(best_ind[:M_local])
        t_scales     = np.asarray(best_ind[M_local:])
        omega_lin_vec = baseline_omega_lin * omega_scales
        t_sec_vec     = baseline_t_sec * t_scales

        omega_lin_full = omega_lin_vec[inverse]
        t_sec_full     = t_sec_vec[inverse]

        best_seq_host = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            omega_lin_full.astype(float),
            t_sec_full.astype(float),
        ])
        np.save(gen_dir / "best_sequence.npy", best_seq_host)

        # Extract scale vectors
        M_local = baseline_omega_lin.shape[0]
        omega_scales = np.asarray(best_ind[:M_local])
        t_scales     = np.asarray(best_ind[M_local:])

        # Build actual Ω and t vectors
        omega_vec = (baseline_omega_lin * omega_scales).tolist()
        time_vec  = (baseline_t_sec * t_scales).tolist()

        meta = {
            "gen": gen,
            "fitness": float(best_ind.fitness.values[0]),
            "score_list": getattr(best_ind, "goods", None),
            "num_pulses": int(base_seq.shape[0]),
            "unique_pairs": unique_pairs.tolist(),
            "omega_vector": omega_vec,     
            "time_vector": time_vec,      
            "omega_scales": omega_scales.tolist(),   
            "time_scales": t_scales.tolist(),        
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
        scales = np.asarray(ind)
        M_local = baseline_omega_lin.shape[0]
        omega_scales = scales[:M_local]
        t_scales     = scales[M_local:]

        omega_lin_vec = baseline_omega_lin * omega_scales
        t_sec_vec     = baseline_t_sec * t_scales

        omega_lin_full = omega_lin_vec[inverse]
        t_sec_full     = t_sec_vec[inverse]

        full_seq = np.column_stack([
            axes.astype(float),
            dns.astype(float),
            omega_lin_full.astype(float),
            t_sec_full.astype(float),
        ])
        np.save(final_dir / f"top{rank}_sequence.npy", full_seq)
        # Extract scale vectors
        scales = np.asarray(ind)
        M_local = baseline_omega_lin.shape[0]
        omega_scales = scales[:M_local]
        t_scales     = scales[M_local:]

        # Compute actual Ω and t vectors
        omega_vec = (baseline_omega_lin * omega_scales).tolist()
        time_vec  = (baseline_t_sec * t_scales).tolist()

        meta = {
            "gen": gen,
            "fitness": float(ind.fitness.values[0]),
            "score_list": getattr(ind, "goods", None),
            "num_pulses": int(base_seq.shape[0]),
            "unique_pairs": unique_pairs.tolist(),
            "omega_vector": omega_vec,   
            "time_vector": time_vec,      
            "omega_scales": omega_scales.tolist(),  
            "time_scales": t_scales.tolist(),        
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
        n_gen=50,
        pop_size=64,
        cx_prob=0.7,
        mut_prob=0.8,
        py_seed=42,
        n_molecules=10_000,
        temp=(25e-6, 25e-6, 25e-6),
        K_max=30,
        n_good=3,
        trap_dets=[-2e3, -1e3, 0.0, 1e3, 2e3],
        base_seq_npy="seq_partial.npy",
        outdir="omega_time_ga_runs",
        omega_scale_min=0.1,
        omega_scale_max=1.5,
        t_scale_min=0.1,
        t_scale_max=5.0,
    )
    run_omega_time_ga(cfg, res)
