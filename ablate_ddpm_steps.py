"""DDPM inference ablation: measures FID at different DDIM step counts.

The model is already trained. We just vary ddim_steps and compute FID for each.

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

def main() -> None:
    set_seed(42)
    print(f"Device: {DEVICE}\n")
    Path("outputs/ablation").mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(CONFIG)
    mc = cfg["model"]
    dc = cfg["diffusion"]
    img_size = cfg["data"]["image_size"]

    model = UNet(
        image_channels=mc["image_channels"],
        base_channels=mc["base_channels"],
        channel_mults=tuple(mc["channel_mults"]),
        attention_resolutions=tuple(mc["attention_resolutions"]),
        num_res_blocks=mc["num_res_blocks"],
        dropout=mc["dropout"],
    ).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=DEVICE,
    )

    sample_shape = (1, mc["image_channels"], img_size, img_size)
    results = {}

    for steps in DDIM_STEPS_LIST:
        print(f"\nddim_steps = {steps}")
        out_dir = f"outputs/ablation/ddpm_steps{steps}/fid_samples"
        save_samples(
            model, out_dir,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            diffusion=diffusion,
            sample_shape=sample_shape,
            ddim_steps=steps,
            eta=0.0,
        )
        fid = compute_fid(REAL_DIR, out_dir)
        print(f"FID (steps={steps}): {fid:.4f}")
        results[str(steps)] = {"ddim_steps": steps, "fid": fid}

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\nDDPM DDIM steps ablation results:")
    for steps in DDIM_STEPS_LIST:
        r = results.get(str(steps))
        if r:
            print(f"steps={steps:<4}  FID: {r['fid']:.4f}")
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
