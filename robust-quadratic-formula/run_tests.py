import subprocess
import os
import math
import numpy as np
import mpmath
import matplotlib.pyplot as plt

# Compile
src = os.path.join(os.path.dirname(__file__), "quadratic.c")
exe = os.path.join(os.path.dirname(__file__), "quadratic")
subprocess.run(["gcc", "-O2", "-o", exe, src, "-lm"], check=True)

mpmath.mp.dps = 50

# delta sweeps from 1 toward 0. x approaches the boundary of the cone and dx becomes shorter (as would happen in IPM convergence).
deltas = [10**(-k) for k in range(16)]

test_cases      = []
true_small_roots = []   # root near 0 (max of the two, since a < 0)

for delta in deltas:
    # x = [sqrt(2), 1-delta, 1-delta], dx = [0.5, 1, 1] * delta
    x0  = math.sqrt(2)
    dx0 = 1 * delta
    dx1 = np.array([2.0, 2.0]) * delta
    a = dx0**2 - np.dot(dx1, dx1)

    x1 = np.array([1 - delta, 1 - delta])
    b  = 2 * (x0 * dx0 - np.dot(x1, dx1))
    c  = x0**2 - np.dot(x1, x1)
    test_cases.append((a, b, c))

    ma, mb, mc = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(c)
    d   = mpmath.sqrt(mb**2 - 4*ma*mc)
    r1  = (-mb + d) / (2*ma)
    r2  = (-mb - d) / (2*ma)
    pos = [r for r in (r1, r2) if r > 0]
    true_small_roots.append(float(min(pos)))
naive_small = []
smart_small = []

for a_val, b_val, c_val in test_cases:
    result = subprocess.run(
        [exe, repr(float(a_val)), repr(float(b_val)), repr(float(c_val))],
        capture_output=True, text=True, check=True,
    )
    nv, sv = map(float, result.stdout.split())
    naive_small.append(nv)
    smart_small.append(sv)

print(f"{'delta':>10}  {'true (small root)':>22} {'naive':>22} {'smart':>22}  {'err_naive':>12} {'err_smart':>12}")
print("-" * 112)
err_naive, err_smart = [], []
for delta, true_r, nv, sv in zip(deltas, true_small_roots, naive_small, smart_small):
    en = abs(nv - true_r) / abs(true_r) if true_r != 0 else abs(nv - true_r)
    es = abs(sv - true_r) / abs(true_r) if true_r != 0 else abs(sv - true_r)
    err_naive.append(en)
    err_smart.append(es)
    print(f"{delta:>10.0e}  {true_r:>22.15g} {nv:>22.15g} {sv:>22.15g}  {en:>12.3e} {es:>12.3e}")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, err_naive, "o-", label="quadratic")
ax.loglog(deltas, err_smart, "s-", label="citardauq")
ax.invert_xaxis()
ax.set_xlabel("delta  (decreasing →  iterate approaches cone boundary)")
ax.set_ylabel("relative error in step length")
ax.set_title("Relative error in step length as cone boundary is approached")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "step_length_error.png"), dpi=200)
plt.show()
