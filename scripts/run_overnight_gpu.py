from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.fid import compute_fid, save_samples, save_real_samples
from src.models.dcgan import Discriminator, Generator
from src.models.ddpm import GaussianDiffusion, UNet
from src.training.train_dcgan import train as _train_dcgan
from src.utils.seed import set_seed

OUT_ROOT = Path("outputs/overnight_safe")
CKPT_ROOT = Path("checkpoints/overnight_safe")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DIR = "outputs/real_64"
N_SAMPLES = 5000
BATCH_SIZE = 16


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def job1_ddpm_ema_ablation() -> None:
    checkpoint = "checkpoints/ddpm/ddpm_epoch0200.pt"
    cfg = load_yaml("configs/ddpm_config.yaml")
    mc, dc = cfg["model"], cfg["diffusion"]
    img_size = cfg["data"]["image_size"]
    sample_shape = (1, mc["image_channels"], img_size, img_size)

    state = torch.load(checkpoint, map_location=DEVICE)

    def build_unet() -> UNet:
        return UNet(
            image_channels=mc["image_channels"],
            base_channels=mc["base_channels"],
            channel_mults=tuple(mc["channel_mults"]),
            attention_resolutions=tuple(mc["attention_resolutions"]),
            num_res_blocks=mc["num_res_blocks"],
            dropout=mc["dropout"],
        ).to(DEVICE)

    model_raw = build_unet()
    model_raw.load_state_dict(state["model"])
    model_raw.eval()

    model_ema = build_unet()
    model_ema.load_state_dict(state["model"])
    if "ema_shadow" in state:
        shadow = state["ema_shadow"]
        with torch.no_grad():
            for name, param in model_ema.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(shadow[name])
    model_ema.eval()

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"], beta_start=dc["beta_start"],
        beta_end=dc["beta_end"], schedule=dc["schedule"], device=DEVICE,
    )

    results_file = OUT_ROOT / "ddpm_ema_ablation_results.json"
    results: dict = {}

    for steps in [10, 25, 50, 100, 200]:
        set_seed(42)
        entry: dict = {"ddim_steps": steps}

        out_raw = str(OUT_ROOT / f"ddpm_steps{steps}" / "fid_raw")
        save_samples(model_raw, out_raw, n_samples=N_SAMPLES, batch_size=BATCH_SIZE,
                     device=DEVICE, diffusion=diffusion, sample_shape=sample_shape,
                     ddim_steps=steps, eta=0.0)
        entry["fid_raw"] = compute_fid(REAL_DIR, out_raw)

        out_ema = str(OUT_ROOT / f"ddpm_steps{steps}" / "fid_ema")
        save_samples(model_ema, out_ema, n_samples=N_SAMPLES, batch_size=BATCH_SIZE,
                     device=DEVICE, diffusion=diffusion, sample_shape=sample_shape,
                     ddim_steps=steps, eta=0.0)
        entry["fid_ema"] = compute_fid(REAL_DIR, out_ema)

        print(f"steps={steps}  raw={entry['fid_raw']:.2f}  ema={entry['fid_ema']:.2f}")
        results[str(steps)] = entry
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


def job2_wgangp_retrain() -> None:
    cfg = load_yaml("configs/dcgan_config.yaml")
    cfg["logging"]["output_dir"] = str(OUT_ROOT / "wgangp_retrained")
    cfg["logging"]["checkpoint_dir"] = str(CKPT_ROOT / "wgangp_retrained")

    _train_dcgan(cfg)

    ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "ckpt_epoch0100.pt"
    mc = cfg["model"]

    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    G.load_state_dict(state["G"])
    G.eval()

    set_seed(42)
    out_dir = OUT_ROOT / "wgangp_retrained" / "fid_samples"
    save_samples(G, str(out_dir), n_samples=N_SAMPLES, batch_size=64,
                 device=DEVICE, z_dim=mc["z_dim"])

    fid = compute_fid(REAL_DIR, str(out_dir))
    print(f"wgangp retrained fid={fid:.2f}")

    result = {
        "model": "wgangp_retrained_safe",
        "fid": fid,
        "epochs": cfg["training"]["epochs"],
        "output_dir": cfg["logging"]["output_dir"],
        "checkpoint_dir": cfg["logging"]["checkpoint_dir"],
    }
    (OUT_ROOT / "wgangp_retrained.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def job3_dcgan_std_multiseed() -> None:
    base_cfg = load_yaml("configs/dcgan_std_corrected_config.yaml")
    seeds = [0, 42, 123]
    results: dict = {}
    results_file = OUT_ROOT / "dcgan_std_multiseed.json"

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        cfg["logging"]["output_dir"] = str(OUT_ROOT / f"dcgan_std_seed{seed}")
        cfg["logging"]["checkpoint_dir"] = str(CKPT_ROOT / f"dcgan_std_seed{seed}")
        _train_dcgan(cfg)

        ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "ckpt_epoch0150.pt"
        mc = cfg["model"]
        G = Generator(
            z_dim=mc["z_dim"], base_channels=mc["g_channels"],
            image_size=cfg["data"]["image_size"],
            image_channels=mc["image_channels"],
        ).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(state["G"])
        G.eval()

        set_seed(seed)
        out_dir = OUT_ROOT / f"dcgan_std_seed{seed}" / "fid_samples"
        save_samples(G, str(out_dir), n_samples=N_SAMPLES, batch_size=64,
                     device=DEVICE, z_dim=mc["z_dim"])
        fid = compute_fid(REAL_DIR, str(out_dir))
        print(f"seed={seed}  fid={fid:.2f}")
        results[str(seed)] = {"seed": seed, "fid": fid}
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fids = [v["fid"] for v in results.values()]
    mean_fid = sum(fids) / len(fids)
    std_fid = (sum((f - mean_fid) ** 2 for f in fids) / len(fids)) ** 0.5
    print(f"dcgan std 3-seed: mean={mean_fid:.2f} std={std_fid:.2f}")
    results["summary"] = {"mean": mean_fid, "std": std_fid}
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    if not Path(REAL_DIR).exists():
        print(f"real_64 not found at {REAL_DIR}, run evaluate_all.py first")
        sys.exit(1)

    print(f"device: {DEVICE}")
    job1_ddpm_ema_ablation()
    job2_wgangp_retrain()
    job3_dcgan_std_multiseed()


if __name__ == "__main__":
    main()
