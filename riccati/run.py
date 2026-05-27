import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

# Problem is
# minimize (1/2)\sum_{k=1}^{T-1} (x_k^\top Q x_k + u_k^\top R u_k) + (1/2)x_T^\top Q_f x_T
# subject to x_{k+1} = Ax_k + Bu_k

def finite_horizon_riccati(A, B, Qf, Q, R, T, x1):
    P = [None for _ in range(T + 1)]
    K = [None for _ in range(T)]

    x = np.zeros((nx, T))
    u = np.zeros((nu, T - 1))
    P[-1] = Qf
    for k in range(T-1, 0, -1):
        K[k] = np.linalg.solve(R + B.T @ P[k+1] @ B, B.T @ P[k+1] @ A)
        P[k] = Q + K[k].T @ R @ K[k] + (A - B @ K[k]).T @ P[k+1] @ (A - B @ K[k])
    
    x[:,0] = x1
    for k in range(T - 1):
        u[:,k] = -K[k+1] @ x[:,k]
        x[:,k+1] = A @ x[:,k] + B @ u[:,k]
    return x, u

def infinite_horizon_riccati(A, B, Qf, Q, R, T, x1):
    P = [None for _ in range(T + 1)]
    K = [None for _ in range(T)]

    x = np.zeros((nx, T))
    u = np.zeros((nu, T - 1))
    P = Qf
    K = None
    for k in range(1000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Pnew = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(Pnew - P, 'fro') < 1e-6:
            print(f"Fixed-point iteration took {k} iterations")
            break
        P = Pnew
    
    x[:,0] = x1
    for k in range(T - 1):
        u[:,k] = -K @ x[:,k]
        x[:,k+1] = A @ x[:,k] + B @ u[:,k]
    return x, u


def finite_horizon_cvxpy(A, B, Qf, Q, R, T, x1):
    nx, nu = B.shape
    x = cp.Variable((nx, T))
    u = cp.Variable((nu, T - 1))
    con = [x[: , 0] == x1]
    obj = 0.5 * cp.quad_form(x[:,-1], Qf)
    for k in range(T - 1):
        con += [x[:, k+1] == A @ x[:,k] + B @ u[:,k]]
        obj += 0.5 * (cp.quad_form(x[:,k], Q) + cp.quad_form(u[:,k], R))
    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve()
    return x.value, u.value

def plot_channels(series, labels, ylabel_prefix, filename):
    n_channels = series[0].shape[0]
    fig, axes = plt.subplots(n_channels, 1, sharex=True, dpi=200)
    axes = np.atleast_1d(axes)
    t = np.arange(series[0].shape[1])

    for i, ax in enumerate(axes):
        for values, label in zip(series, labels):
            ax.plot(t, values[i, :], label=label)
        ax.set_ylabel(f"{ylabel_prefix}[{i}]")
        ax.grid(True)

    axes[-1].set_xlabel("time step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# Linearized discrete-time vehicle model around straight-line motion.
# State: [x position, y position, speed error, heading error, yaw-rate error].
# Input: [longitudinal acceleration, yaw acceleration].
dt = 0.1
v_ref = 10.0
nx, nu = 5, 2
A = np.array([
    [1.0, 0.0, dt,  0.0,        0.0],
    [0.0, 1.0, 0.0, v_ref * dt, 0.0],
    [0.0, 0.0, 1.0, 0.0,        0.0],
    [0.0, 0.0, 0.0, 1.0,        dt],
    [0.0, 0.0, 0.0, 0.0,        1.0],
])
B = np.array([
    [0.5 * dt**2, 0.0],
    [0.0,         0.0],
    [dt,          0.0],
    [0.0,         0.5 * dt**2],
    [0.0,         dt],
])
Q = np.diag([20.0, 20.0, 2.0, 10.0, 1.0])
R = np.diag([0.5, 0.2])
Qf = np.diag([100.0, 100.0, 10.0, 50.0, 5.0])
T  = 15
x1 = np.array([5.0, -2.0, 1.0, 0.2, -0.1])
xr, ur = finite_horizon_riccati(A, B, Qf, Q, R, T, x1)
xir, uir = infinite_horizon_riccati(A, B, Qf, Q, R, T, x1)
xc, uc = finite_horizon_cvxpy(A, B, Qf, Q, R, T, x1)

print("||xir-xc||^2: ", np.linalg.norm(xir - xc, 'fro'))
print("||uir-uc||^2: ", np.linalg.norm(uir - uc, 'fro'))
print("||xr-xc||^2: ", np.linalg.norm(xr - xc, 'fro'))
print("||ur-uc||^2: ", np.linalg.norm(ur - uc, 'fro'))

labels = ["finite horizon riccati", "infinite horizon riccati", "finite horizon cvxpy"]
plot_channels([ur, uir, uc], labels, "u", "control.pdf")
plot_channels([xr, xir, xc], labels, "x", "state.pdf")