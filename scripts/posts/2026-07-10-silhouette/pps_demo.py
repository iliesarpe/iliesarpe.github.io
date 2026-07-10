"""
Illustrates the intuition behind PPS (Probability Proportional to Size) sampling
[Chechik, Cohen, Kaplan 2015], as used in SILH-PPS-ALL for silhouette estimation.

Setup: a cluster C of 101 points -- 100 "inliers" uniform in a ball of radius 1,
plus a single outlier at distance 10 from the ball's center. We compute, for each
point e, the PPS sampling probability p_e that Algorithm 1 (silh-pps-all) would
assign it, and show that the outlier gets a dramatically higher probability than
any inlier -- which is exactly why uniform sampling fails on this kind of instance
(Theorem 2) while PPS sampling succeeds (Theorem 1).
"""
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\RequirePackage[T1]{fontenc} \RequirePackage[tt=false, type1=true]{libertine} \RequirePackage[varqu]{zi4} \RequirePackage[libertine]{newtxmath} \RequirePackage{xfrac} \RequirePackage{amsmath}')
rng = np.random.default_rng(0)
clrs = sns.color_palette("tab10")

# --- build the cluster ---
n_inliers = 100
radius = 1.0
outlier_dist = 10.0

angles = rng.uniform(0, 2 * np.pi, n_inliers)
radii = radius * np.sqrt(rng.uniform(0, 1, n_inliers))  # uniform over the disk
inliers = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
outlier = np.array([[outlier_dist, 0.0]])

points = np.vstack([inliers, outlier])
n = points.shape[0]
outlier_idx = n - 1

# --- W_C(e) = sum of distances from e to every other point in C (Eq. 4) ---
D = squareform(pdist(points)) # compute distances for each point
W = D.sum(axis=1)

# --- Algorithm 1, Phase 1: real (randomized) seed sample F^0, not a shortcut ---
# F^0 is obtained by including each point independently with probability
# (2/|C|) * ln(5k/delta); with this sample the algorithm is *proven* (Lemma 1)
# to contain a well-positioned element with probability >= 1 - delta/(5k).
delta = 0.1
k = 4     # nominal number of clusters in the full clustering (only enters via log(5k/delta))
t = 15    # expected final sample size (same role as before)

q0 = min(1.0, (2.0 / n) * np.log(5 * k / delta))
seed_mask = rng.random(n) < q0
f0_idx = np.flatnonzero(seed_mask)
if f0_idx.size == 0:
    # extremely unlikely for our q0, but Phase 1 offers no guarantee if it happens
    raise RuntimeError("empty seed sample F^0 -- rerun with a different seed")

# --- gamma_e = max{1/|C|, max_{e_bar in F^0} d(e, e_bar) / W(e_bar)}  (Eq. 6) ---
gamma = np.maximum(1.0 / n, (D[:, f0_idx] / W[f0_idx]).max(axis=1))

# --- p_e = min(1, t * gamma_e) ---
p = np.minimum(1.0, t * gamma)

# --- Phase 1's *final* sample F, drawn with these probabilities ---
final_mask = rng.random(n) < p
final_idx = np.flatnonzero(final_mask)

print(f"seed sample F^0: {f0_idx.size} points (q0={q0:.4f}), outlier included: {outlier_idx in f0_idx}")
print(f"final sample F: {final_idx.size} points, outlier included: {outlier_idx in final_idx}")
print(f"inlier p_e:  min={p[:n_inliers].min():.5f}  max={p[:n_inliers].max():.5f}  mean={p[:n_inliers].mean():.5f}")
print(f"outlier p_e: {p[outlier_idx]:.5f}")
print(f"outlier / mean-inlier ratio: {p[outlier_idx] / p[:n_inliers].mean():.1f}x")

# --- plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5))

size = lambda pe: 15 + 300 * pe  # sqrt-ish visual scale, kept small so legend stays readable
ax1.scatter(points[:n_inliers, 0], points[:n_inliers, 1],
            s=size(p[:n_inliers]), color=clrs[1], alpha=0.6, label="close to origin")
ax1.scatter(points[outlier_idx, 0], points[outlier_idx, 1],
            s=size(p[outlier_idx]), color=clrs[2], alpha=0.8, label="outlier")
ax1.scatter(points[final_idx, 0], points[final_idx, 1],
            s=size(p[final_idx]) + 40, facecolors="none", edgecolors="black",
            linewidths=1.2, label="selected in the sample")

ax1.set_title("Points in $\mathbb{R}^2$ with size proportional to PPS probability $p_e$")
#ax1.set_aspect("equal")
ax1.legend(loc="best", markerscale=0.5, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")
ax1.grid(linestyle='--', alpha=0.7)

dist_from_center = np.linalg.norm(points, axis=1)
ax2.scatter(dist_from_center[:n_inliers], p[:n_inliers], color=clrs[1], label="close to origin")
ax2.scatter(dist_from_center[outlier_idx], p[outlier_idx], color=clrs[2], s=80, label="outlier")
ax2.set_yscale("log")
ax2.set_xlabel("Distance from origin")
ax2.set_ylabel("Sampling probability $p_e$ (log scale)")
ax2.set_title("PPS probability vs. distance from center")
ax2.legend(loc="best")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(False)
ax2.grid(linestyle='--', alpha=0.7)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "pps_probabilities.png")
fig.savefig(out_path, dpi=200)
print(f"saved plot to {out_path}")
