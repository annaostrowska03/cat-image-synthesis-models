"""DDPM inference ablation: measures FID at different DDIM step counts.

Computes FID for both raw model weights and EMA weights at each step count,
producing a 2D (steps × weights) ablation table.

Usage:
    python ablate_ddpm_steps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.evaluation.fid import compute_fid, save_samples
from src.models.ddpm import GaussianDiffusion, UNet
from src.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DIR = "outputs/real_64"
CHECKPOINT = "checkpoints/ddpm/ddpm_epoch0200.pt"
CONFIG = "configs/ddpm_config.yaml"
RESULTS_FILE = "outputs/ablation/ddpm_steps_results.json"

for _required in [REAL_DIR, CHECKPOINT, CONFIG]:
    if not Path(_required).exists():
        raise FileNotFoundError(
            f"Required file/directory not found: '{_required}'. "
            "Ensure the DDPM model is trained and real samples are generated before running ablation."
        )

DDIM_STEPS_LIST = [10, 25, 50, 100, 200]
N_SAMPLES = 5000
BATCH_SIZE = 16  # keep at 16 for 6 GB VRAM

def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(mc: dict) -> UNet:
    return UNet(
        image_channels=mc["image_channels"],
        base_channels=mc["base_channels"],
        channel_mults=tuple(mc["channel_mults"]),
        attention_resolutions=tuple(mc["attention_resolutions"]),
        num_res_blocks=mc["num_res_blocks"],
        dropout=mc["dropout"],
    ).to(DEVICE)

def main() -> None:
    set_seed(42)
    print(f"Device: {DEVICE}\n")
    Path("outputs/ablation").mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(CONFIG)
    mc = cfg["model"]
    dc = cfg["diffusion"]
    img_size = cfg["data"]["image_size"]
    sample_shape = (1, mc["image_channels"], img_size, img_size)

    state = torch.load(CHECKPOINT, map_location=DEVICE)

    # Raw weights
    model_raw = build_model(mc)
    model_raw.load_state_dict(state["model"])
    model_raw.eval()

    # EMA weights
    model_ema = build_model(mc)
    model_ema.load_state_dict(state["model"])
    if "ema_shadow" in state:
        shadow = state["ema_shadow"]
        with torch.no_grad():
            for name, param in model_ema.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(shadow[name])
    model_ema.eval()

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=DEVICE,
    )

    results: dict[str, dict] = {}

    for steps in DDIM_STEPS_LIST:
        print(f"\nddim_steps = {steps}")
        entry: dict = {"ddim_steps": steps}

        out_raw = f"outputs/ablation/ddpm_steps{steps}/fid_raw"
        save_samples(
            model_raw, out_raw,
            n_samples=N_SAMPLES, batch_size=BATCH_SIZE, device=DEVICE,
            diffusion=diffusion, sample_shape=sample_shape,
            ddim_steps=steps, eta=0.0,
        )
        entry["fid_raw"] = compute_fid(REAL_DIR, out_raw)
        print(f"  FID raw  (steps={steps}): {entry['fid_raw']:.4f}")

        # EMA
        out_ema = f"outputs/ablation/ddpm_steps{steps}/fid_ema"
        save_samples(
            model_ema, out_ema,
            n_samples=N_SAMPLES, batch_size=BATCH_SIZE, device=DEVICE,
            diffusion=diffusion, sample_shape=sample_shape,
            ddim_steps=steps, eta=0.0,
        )
        entry["fid_ema"] = compute_fid(REAL_DIR, out_ema)
        print(f"  FID EMA  (steps={steps}): {entry['fid_ema']:.4f}")

        results[str(steps)] = entry
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\nDDPM DDIM steps ablation (steps × weights)")
    print(f"{'Steps':>6}  {'FID raw':>10}  {'FID EMA':>10}  {'Speedup':>8}")
    speedups = {10: "100×", 25: "40×", 50: "20×", 100: "10×", 200: "5×"}
    for steps in DDIM_STEPS_LIST:
        r = results.get(str(steps))
        if r:
            print(f"{steps:>6}  {r['fid_raw']:>10.2f}  {r['fid_ema']:>10.2f}  {speedups[steps]:>8}")
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
