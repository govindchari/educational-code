# Riccati LQR Example

This repository contains a compact Python example for solving a discrete-time
linear quadratic regulator (LQR) problem with Riccati recursions and comparing
the result against a CVXPY optimization solve.

The example uses a linearized vehicle model around straight-line motion, with
state

```text
[x position, y position, speed error, heading error, yaw-rate error]
```

and control input

```text
[longitudinal acceleration, yaw acceleration]
```

## What It Does

`run.py` computes trajectories for the same initial condition using:

- finite-horizon Riccati recursion
- infinite-horizon Riccati fixed-point iteration
- finite-horizon direct optimization with CVXPY

It then prints Frobenius-norm differences between the Riccati trajectories and
the CVXPY solution, and writes comparison plots to:

- `control.pdf`
- `state.pdf`

## Problem Form

The finite-horizon problem is:

```math
\begin{aligned}
\underset{x_1, \ldots, x_T,\; u_1, \ldots, u_{T-1}}{\text{minimize}} \quad
& \frac{1}{2} \sum_{k=1}^{T-1} \left(x_k^\top Q x_k + u_k^\top R u_k\right)
  + \frac{1}{2} x_T^\top Q_f x_T \\
\text{subject to} \quad
& x_{k+1} = A x_k + B u_k, \qquad k = 1, \ldots, T-1, \\
& x_1 = x_{\mathrm{init}}.
\end{aligned}
```

## Notes

The finite-horizon Riccati solution should be identical to the CVXPY solution.
The infinite-horizon controller is computed from a fixed-point Riccati iteration. For long horizons, the early finite-horizon Riccati/CVXPY feedback gains, controls, and states should approach those from the infinite-horizon controller. Differences are mainly expected near the terminal end of the horizon, where the terminal cost influences the finite-horizon solution.

This convergence assumes the usual LQR conditions: the system is stabilizable,
the cost is well posed, and the state and input weights are positive
semidefinite and positive definite as required.

If the terminal cost `Qf` is chosen to equal the stabilizing infinite-horizon
Riccati solution `P_infty`, then the finite-horizon Riccati recursion is
stationary immediately. The finite-horizon backward recursion is

```math
\begin{aligned}
K_k &= (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A, \\
P_k &= Q + K_k^\top R K_k + (A - B K_k)^\top P_{k+1} (A - B K_k).
\end{aligned}
```

Because `P_infty` is a fixed point of this recursion, setting `P_T = Qf =
P_infty` gives `P_k = P_infty` and `K_k = K_infty` for every timestep.
Therefore the finite-horizon Riccati solution matches the infinite-horizon
solution exactly, up to numerical error.

Intuitively, choosing `Qf = P_infty` tells the finite-horizon problem that after
the horizon ends, the remaining infinite-horizon cost is being represented
exactly.
