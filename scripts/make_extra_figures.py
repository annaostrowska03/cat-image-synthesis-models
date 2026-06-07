"""Generate three extra figures for report and presentation:
  1. outputs/compute_quality.png  -- FID vs training time scatter
  2. outputs/dcgan_progression.png -- DCGAN samples across epochs
  3. outputs/real_vs_generated.png -- real cats vs best generated (DDPM)

Run from the project root:
    python scripts/make_extra_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Compute vs quality scatter
models = [
    ("DCGAN\n(original)", 2.0, 221.51, "#b5838d"),
    ("DCGAN\n(std, fixed)", 3.0, 98.99, "#e07a5f"),
    ("DCGAN\n(WGAN-GP, fixed)", 4.5, 158.48, "#f2cc8f"),
    ("VQ-VAE\n+PixelCNN", 4.0, 187.54, "#81b29a"),
    ("DDPM\n+EMA", 12.0, 24.36, "#3d405b"),
]

fig, ax = plt.subplots(figsize=(7, 4.2))
for label, hours, fid, color in models:
    ax.scatter(hours, fid, s=160, color=color, zorder=3, edgecolors="white", linewidths=0.8)
    # nudge labels to avoid overlap
    xoff = 0.25 if hours < 11 else -0.35
    yoff = 6 if fid > 100 else -16
    ha = "left" if xoff > 0 else "right"
    ax.annotate(label, (hours, fid), xytext=(hours + xoff, fid + yoff),
                fontsize=8, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7))

ax.set_xlabel("Approximate training time (hours)", fontsize=10)
ax.set_ylabel("FID (↓ better)", fontsize=10)
ax.set_title("Quality vs. Compute Trade-off", fontsize=11, fontweight="bold")
ax.set_xlim(0, 14)
ax.set_ylim(-10, 260)
ax.axhline(y=100, color="#ccc", lw=0.8, ls="--")
ax.grid(True, alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "compute_quality.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/compute_quality.png")

# DCGAN training progression
epochs = [10, 30, 50, 70, 100]
paths = [OUT / f"dcgan/samples_epoch{e:04d}.png" for e in epochs]
missing = [p for p in paths if not p.exists()]
if missing:
    print(f"[WARN] missing files: {missing}")
    paths = [p for p in paths if p.exists()]
    epochs = [int(p.stem.split("epoch")[1]) for p in paths]

n = len(paths)
fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0))
for ax, path, ep in zip(axes, paths, epochs):
    img = mpimg.imread(str(path))
    ax.imshow(img)
    ax.set_title(f"Epoch {ep}", fontsize=10, fontweight="bold")
    ax.axis("off")
fig.suptitle("DCGAN Training Progression (original, n_critic=5)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "dcgan_progression.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/dcgan_progression.png")

# Real vs generated comparison
real_path = OUT / "data_samples_64.png"
dcgan_path = OUT / "dcgan/samples_epoch0100.png"
vqvae_path = OUT / "vqvae/prior_samples.png"
ddpm_path = OUT / "ddpm/samples_epoch0200.png"

rows = [
    ("Real cats", real_path),
    ("DCGAN  FID 221.5", dcgan_path),
    ("VQ-VAE  FID 187.5", vqvae_path),
    ("DDPM  FID 24.4", ddpm_path),
]
rows = [(lbl, p) for lbl, p in rows if p.exists()]

fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.4 * len(rows)))
if len(rows) == 1:
    axes = [axes]
for ax, (label, path) in zip(axes, rows):
    img = mpimg.imread(str(path))
    ax.imshow(img)
    ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=0,
                  labelpad=90, va="center")
    ax.axis("off")
fig.suptitle("Real vs. Generated Samples", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "real_vs_generated.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved outputs/real_vs_generated.png")
