#!/usr/bin/env python3
"""Benchmark repeated LQR KKT solves with qdldl and JAX."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qdldl
import scipy.sparse as sp
import scipy.sparse.linalg
from spineax.cudss.solver import CuDSSSolver


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LqrProblem:
    name: str
    plot_prefix: str
    a_mat: np.ndarray
    b_mat: np.ndarray
    q_base: np.ndarray
    r_base: np.ndarray


def double_integrator(dt: float) -> tuple[np.ndarray, np.ndarray]:
    a_mat = np.array([[1.0, dt], [0.0, 1.0]])
    b_mat = np.array([[0.5 * dt * dt], [dt]])
    return a_mat, b_mat


def larger_lqr_dynamics(dt: float) -> tuple[np.ndarray, np.ndarray]:
    nx = 16
    nu = 8
    a_mat = 0.92 * np.eye(nx)
    for i in range(nx - 1):
        a_mat[i, i + 1] = 0.03
        a_mat[i + 1, i] = -0.02
    for i in range(nx - 4):
        a_mat[i, i + 4] = 0.01

    b_mat = np.zeros((nx, nu))
    for i in range(nx):
        b_mat[i, i % nu] = 0.2 * dt
        b_mat[i, (i + 3) % nu] += 0.05 * dt
    return a_mat, b_mat


def make_problems(dt: float) -> list[LqrProblem]:
    a_1d, b_1d = double_integrator(dt)
    a_large, b_large = larger_lqr_dynamics(dt)
    return [
        LqrProblem(
            name="Double integrator",
            plot_prefix="double_integrator",
            a_mat=a_1d,
            b_mat=b_1d,
            q_base=np.diag([1.0, 0.1]),
            r_base=np.array([[0.01]]),
        ),
        LqrProblem(
            name="Coupled system",
            plot_prefix="larger_coupled_system",
            a_mat=a_large,
            b_mat=b_large,
            q_base=np.diag(np.linspace(1.0, 0.2, a_large.shape[0])),
            r_base=0.01 * np.eye(b_large.shape[1]),
        ),
    ]


def x_slice(t: int, nx: int) -> slice:
    return slice(t * nx, (t + 1) * nx)


def u_slice(t: int, nx: int, nu: int, horizon: int) -> slice:
    offset = (horizon + 1) * nx
    return slice(offset + t * nu, offset + (t + 1) * nu)


def build_lqr_kkt(
    problem: LqrProblem,
    horizon: int,
    sigma: float,
    rho_inv: float,
    q_scale: float,
    r_scale: float,
) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    a_mat = problem.a_mat
    b_mat = problem.b_mat
    nx = a_mat.shape[0]
    nu = b_mat.shape[1]

    q_mat = q_scale * problem.q_base
    q_terminal = 10.0 * q_mat
    r_mat = r_scale * problem.r_base

    n_vars = (horizon + 1) * nx + horizon * nu
    n_constraints = (horizon + 1) * nx

    p_blocks: list[sp.csc_matrix] = []
    for _ in range(horizon):
        p_blocks.append(sp.csc_matrix(q_mat))
    p_blocks.append(sp.csc_matrix(q_terminal))
    for _ in range(horizon):
        p_blocks.append(sp.csc_matrix(r_mat))
    p_mat = sp.block_diag(p_blocks, format="csc")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_block(row0: int, col0: int, block: np.ndarray) -> None:
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                val = block[i, j]
                if val != 0.0:
                    rows.append(row0 + i)
                    cols.append(col0 + j)
                    data.append(float(val))

    add_block(0, x_slice(0, nx).start, np.eye(nx))

    for t in range(horizon):
        row0 = (t + 1) * nx
        add_block(row0, x_slice(t + 1, nx).start, np.eye(nx))
        add_block(row0, x_slice(t, nx).start, -a_mat)
        add_block(row0, u_slice(t, nx, nu, horizon).start, -b_mat)

    c_mat = sp.coo_matrix((data, (rows, cols)), shape=(n_constraints, n_vars)).tocsc()

    k_top_left = p_mat + sigma * sp.eye(n_vars, format="csc")
    k_bottom_right = -rho_inv * sp.eye(n_constraints, format="csc")
    kkt = sp.bmat(
        [[k_top_left, c_mat.T], [c_mat, k_bottom_right]],
        format="csc",
    )
    return kkt, c_mat


def build_rhs_batch(problem: LqrProblem, horizon: int, num_problems: int) -> np.ndarray:
    nx = problem.a_mat.shape[0]
    nu = problem.b_mat.shape[1]
    n_vars = (horizon + 1) * nx + horizon * nu
    n_constraints = (horizon + 1) * nx
    rhs = np.zeros((num_problems, n_vars + n_constraints))

    base_x0 = np.linspace(1.0, -1.0, nx)
    perturb_x0 = np.linspace(-0.5, 0.5, nx)
    scales = np.linspace(-5.0, 5.0, num_problems)
    rhs[:, n_vars : n_vars + nx] = base_x0 + scales[:, None] * perturb_x0
    return rhs


def build_kkt_batch(
    problem: LqrProblem,
    horizon: int,
    num_problems: int,
    sigma: float,
    rho_inv: float,
) -> tuple[list[sp.csc_matrix], np.ndarray, np.ndarray, np.ndarray, sp.csc_matrix]:
    kkt_mats = []
    csr_data = []
    csr_indices = None
    csr_indptr = None
    c_mat = None

    q_scales = np.linspace(0.5, 2.0, num_problems)
    r_scales = np.linspace(2.0, 0.5, num_problems)
    for q_scale, r_scale in zip(q_scales, r_scales):
        kkt, c_mat = build_lqr_kkt(problem, horizon, sigma, rho_inv, q_scale, r_scale)
        kkt_csr = kkt.tocsr()
        kkt_mats.append(kkt)
        csr_data.append(kkt_csr.data)

        if csr_indices is None:
            csr_indices = kkt_csr.indices
            csr_indptr = kkt_csr.indptr
        elif not (
            np.array_equal(csr_indices, kkt_csr.indices)
            and np.array_equal(csr_indptr, kkt_csr.indptr)
        ):
            raise ValueError("KKT matrices must share a sparsity pattern for JAX CSR batching")

    if c_mat is None or csr_indices is None or csr_indptr is None:
        raise ValueError("num_problems must be positive")
    return kkt_mats, np.stack(csr_data), csr_indices, csr_indptr, c_mat


def residual_inf_norm(
    kkt_mats: list[sp.csc_matrix],
    x_batch: np.ndarray,
    rhs_batch: np.ndarray,
) -> float:
    max_residual = 0.0
    for kkt, sol, rhs in zip(kkt_mats, x_batch, rhs_batch):
        residual = kkt @ sol - rhs
        max_residual = max(max_residual, float(np.linalg.norm(residual, ord=np.inf)))
    return max_residual


def time_qdldl(kkt_mats: list[sp.csc_matrix], rhs_batch: np.ndarray) -> tuple[np.ndarray, float]:
    solutions = []

    start = time.perf_counter()
    for kkt, rhs in zip(kkt_mats, rhs_batch):
        solver = qdldl.Solver(kkt)
        solutions.append(solver.solve(rhs.copy()))
    elapsed = time.perf_counter() - start

    return np.vstack(solutions), elapsed


def time_scipy_serial(kkt_mats: list[sp.csc_matrix], rhs_batch: np.ndarray) -> tuple[np.ndarray, float]:
    solutions = []

    start = time.perf_counter()
    for kkt, rhs in zip(kkt_mats, rhs_batch):
        solutions.append(scipy.sparse.linalg.spsolve(kkt, rhs))
    elapsed = time.perf_counter() - start

    return np.vstack(solutions), elapsed


def time_spineax_batched(
    csr_data_batch: np.ndarray,
    csr_indices: np.ndarray,
    csr_indptr: np.ndarray,
    rhs_batch: np.ndarray,
) -> tuple[np.ndarray, float]:
    data_jax = jnp.asarray(csr_data_batch)
    indices_jax = jnp.asarray(csr_indices)
    indptr_jax = jnp.asarray(csr_indptr)
    rhs_jax = jnp.asarray(rhs_batch)

    device_id = 0
    matrix_type_general = 0
    matrix_view_full = 0

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
            solver = CuDSSSolver(
                indptr_jax,
                indices_jax,
                device_id,
                matrix_type_general,
                matrix_view_full,
            )
            solve_batch = jax.jit(jax.vmap(lambda rhs, data: solver(rhs, data)[0]))
            solve_batch(rhs_jax, data_jax).block_until_ready()

    start = time.perf_counter()
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            solutions = solve_batch(rhs_jax, data_jax)
            solutions.block_until_ready()
    elapsed = time.perf_counter() - start

    return np.asarray(solutions), elapsed


def time_spineax_scan(
    csr_data_batch: np.ndarray,
    csr_indices: np.ndarray,
    csr_indptr: np.ndarray,
    rhs_batch: np.ndarray,
) -> tuple[np.ndarray, float]:
    data_jax = jnp.asarray(csr_data_batch)
    indices_jax = jnp.asarray(csr_indices)
    indptr_jax = jnp.asarray(csr_indptr)
    rhs_jax = jnp.asarray(rhs_batch)

    device_id = 0
    matrix_type_general = 0
    matrix_view_full = 0

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
            solver = CuDSSSolver(
                indptr_jax,
                indices_jax,
                device_id,
                matrix_type_general,
                matrix_view_full,
            )
            def scan_solve(_: None, inputs: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
                rhs, data = inputs
                return None, solver(rhs, data)[0]

            solve_scan = jax.jit(lambda rhs, data: jax.lax.scan(scan_solve, None, (rhs, data))[1])
            solve_scan(rhs_jax, data_jax).block_until_ready()

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            start = time.perf_counter()
            solutions = solve_scan(rhs_jax, data_jax)
            solutions.block_until_ready()
    elapsed = time.perf_counter() - start

    return np.asarray(solutions), elapsed


def plot_timings(
    results: dict[str, list[float]],
    batch_sizes: list[int],
    output_path: str,
    ylabel: str,
    title: str,
) -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, times in results.items():
        ax.plot(batch_sizes, times, marker="o", linewidth=2, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(size) for size in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_batch_size(
    problem: LqrProblem,
    horizon: int,
    num_problems: int,
    sigma: float,
    rho_inv: float,
) -> tuple[dict[str, float], dict[str, float]]:
    kkt_mats, csr_data_batch, csr_indices, csr_indptr, _ = build_kkt_batch(
        problem,
        horizon,
        num_problems,
        sigma,
        rho_inv,
    )
    rhs_batch = build_rhs_batch(problem, horizon, num_problems)

    qdldl_sol, qdldl_elapsed = time_qdldl(kkt_mats, rhs_batch)
    scipy_sol, scipy_elapsed = time_scipy_serial(kkt_mats, rhs_batch)
    spineax_sol, spineax_elapsed = time_spineax_batched(
        csr_data_batch,
        csr_indices,
        csr_indptr,
        rhs_batch,
    )
    spineax_scan_sol, spineax_scan_elapsed = time_spineax_scan(
        csr_data_batch,
        csr_indices,
        csr_indptr,
        rhs_batch,
    )

    timings_ms = {
        "qdldl serial": qdldl_elapsed * 1e3,
        "scipy serial": scipy_elapsed * 1e3,
        "spineax serial": spineax_scan_elapsed * 1e3,
        "spineax batch": spineax_elapsed * 1e3,
    }
    residuals = {
        "qdldl serial": residual_inf_norm(kkt_mats, qdldl_sol, rhs_batch),
        "scipy serial": residual_inf_norm(kkt_mats, scipy_sol, rhs_batch),
        "spineax serial": residual_inf_norm(kkt_mats, spineax_scan_sol, rhs_batch),
        "spineax batch": residual_inf_norm(kkt_mats, spineax_sol, rhs_batch),
        "max |qdldl - scipy|": float(np.max(np.abs(qdldl_sol - scipy_sol))),
        "max |qdldl - spineax scan|": float(np.max(np.abs(qdldl_sol - spineax_scan_sol))),
        "max |qdldl - spineax|": float(np.max(np.abs(qdldl_sol - spineax_sol))),
    }
    return timings_ms, residuals


def run_problem_sweep(
    problem: LqrProblem,
    horizon: int,
    batch_sizes: list[int],
    sigma: float,
    rho_inv: float,
) -> None:
    total_plot_path = f"{problem.plot_prefix}_total_solve_times.png"
    per_problem_plot_path = f"{problem.plot_prefix}_solve_time_per_problem.png"

    print(f"problem: {problem.name}")
    print(f"states: {problem.a_mat.shape[0]}")
    print(f"controls: {problem.b_mat.shape[1]}")
    print(f"horizon: {horizon}")
    print(f"batch sizes: {batch_sizes}")
    print("P variation: q_scale in [0.5, 2.0], r_scale in [2.0, 0.5]")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax_gpu_devices}")
    print()

    results = {
        "qdldl serial": [],
        "scipy serial": [],
        "spineax serial": [],
        "spineax batch": [],
    }
    all_residuals: dict[int, dict[str, float]] = {}

    print(
        f"{'batch':>6} {'qdldl ms':>12} {'scipy ms':>12} "
        f"{'spineax serial ms':>18} {'spineax batch ms':>17} {'batch residual':>15}"
    )
    print("-" * 85)
    for batch_size in batch_sizes:
        timings_ms, residuals = run_batch_size(problem, horizon, batch_size, sigma, rho_inv)
        for name in results:
            results[name].append(timings_ms[name])
        all_residuals[batch_size] = residuals

        print(
            f"{batch_size:6d} "
            f"{timings_ms['qdldl serial']:12.3f} "
            f"{timings_ms['scipy serial']:12.3f} "
            f"{timings_ms['spineax serial']:18.3f} "
            f"{timings_ms['spineax batch']:17.3f} "
            f"{residuals['spineax batch']:15.3e}"
        )

    per_problem_results = {
        name: [elapsed_ms / batch_size for elapsed_ms, batch_size in zip(times, batch_sizes)]
        for name, times in results.items()
    }

    plot_timings(
        results,
        batch_sizes,
        total_plot_path,
        "Total solve time (ms)",
        f"Solvetime vs Batch Size {problem.name.lower()}",
    )
    plot_timings(
        per_problem_results,
        batch_sizes,
        per_problem_plot_path,
        "Solve time per problem (ms)",
        f"Solve Time per Problem vs Batch Size {problem.name.lower()}",
    )
    print()
    print(f"saved total-time plot: {total_plot_path}")
    print(f"saved per-problem plot: {per_problem_plot_path}")
    print(
        "max residuals: "
        f"qdldl={max(r['qdldl serial'] for r in all_residuals.values()):.3e}, "
        f"scipy={max(r['scipy serial'] for r in all_residuals.values()):.3e}, "
        f"spineax_serial={max(r['spineax serial'] for r in all_residuals.values()):.3e}, "
        f"spineax_batch={max(r['spineax batch'] for r in all_residuals.values()):.3e}"
    )
    print(
        "max solution diffs vs qdldl: "
        f"scipy={max(r['max |qdldl - scipy|'] for r in all_residuals.values()):.3e}, "
        f"spineax_serial={max(r['max |qdldl - spineax scan|'] for r in all_residuals.values()):.3e}, "
        f"spineax_batch={max(r['max |qdldl - spineax|'] for r in all_residuals.values()):.3e}"
    )
    print()


def main() -> None:
    horizon = 15
    batch_sizes = [2**i for i in range(1, 13)]
    dt = 0.1
    sigma = 1e-6
    rho_inv = 1e-6

    global jax_gpu_devices
    jax_gpu_devices = jax.devices("gpu")
    if not jax_gpu_devices:
        raise RuntimeError(
            "JAX did not initialize a GPU backend. Install CUDA-enabled JAX, "
            'for example: conda run -n batch python -m pip install -U "jax[cuda13]"'
        )

    for problem in make_problems(dt):
        run_problem_sweep(problem, horizon, batch_sizes, sigma, rho_inv)


if __name__ == "__main__":
    main()
