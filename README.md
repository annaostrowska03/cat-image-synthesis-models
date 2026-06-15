# Cat Image Synthesis with Generative Models

This project explores and compares generative modeling approaches for cat image synthesis, including GAN-based, diffusion-based, and VQ-VAE-based methods.

The goal was to understand how different generative architectures model image distributions, how training stability affects output quality, and how learned latent spaces can be analyzed through interpolation and reconstruction experiments.

Author: [Anna Ostrowska](https://github.com/annaostrowska03)
  
## Models implemented

- DCGAN / WGAN-GP for adversarial image generation
- DDPM with DDIM sampling for diffusion-based generation
- VQ-VAE with EMA codebook updates and PixelCNN prior

## Evaluation and analysis

The project includes:
- generated image samples,
- reconstruction and interpolation analysis,
- FID-based evaluation,
- comparison of training behavior across architectures.

## Why this project matters

This project helped me build a deeper understanding of modern generative modeling beyond using pretrained APIs: training dynamics, sampling, latent representations, and quantitative evaluation of generated images.

## Architectures
The project implements and compares the following generative approaches:
* **Generative Adversarial Networks (GANs):** DCGAN with WGAN-GP gradient penalty for training stability.
* **Denoising Diffusion Probabilistic Models (DDPM):** UNet-based noise predictor with DDIM sampling for fast inference.
* **Variational Autoencoders (VAEs):** VQ-VAE with EMA codebook updates and a PixelCNN autoregressive prior.

## Dataset & Preprocessing
* **Primary Dataset:** [Cat Dataset](https://www.kaggle.com/datasets/crawford/cat-dataset): 9,997 images across 7 subfolders (`CAT_00`-`CAT_06`).
* **Exploratory Extension:** [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/) to analyze class-specific feature separation.
* **Preprocessing:** Images are center-cropped and resized to 64×64, then normalized to `[-1, 1]`.

To download the dataset:
```bash
python -m kaggle datasets download crawford/cat-dataset -p data/ --unzip
# or using the full path on Windows:
# & "$env:APPDATA\Python\Python3XX\Scripts\kaggle.exe" datasets download crawford/cat-dataset -p data/ --unzip
```

## Repository Structure
```
src/
  data/           dataset.py, preprocessing.py
  models/         dcgan.py, vqvae.py, ddpm.py
  training/       train_dcgan.py, train_vqvae.py, train_ddpm.py
  evaluation/     fid.py, interpolation.py
  utils/          seed.py, visualization.py
notebooks/        01_data_exploration … 05_comparison_and_interpolation
configs/          dcgan_config.yaml, vqvae_config.yaml, ddpm_config.yaml
evaluate_all.py           # FID + interpolation for all three models
evaluate_cats_and_dogs.py # FID experiment on mixed dataset
ablate_dcgan.py           # DCGAN hyperparameter ablation
ablate_ddpm_steps.py      # DDIM inference steps ablation
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Install uv** (once, system-wide):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install the project** (creates `.venv` automatically):
```bash
uv sync
```

This installs CUDA-enabled PyTorch (cu128 by default). To use a different CUDA version,
edit the `[tool.uv.sources]` index in `pyproject.toml` (e.g. `cu121`, `cu124`, `cpu`).

**Optional extras** (Jupyter notebooks):
```bash
uv sync --extra notebooks
```

## Reproducing Results

**Download data** (Kaggle API required):
```bash
uv run python -m kaggle datasets download crawford/cat-dataset -p data/ --unzip
```

**Train models** (run in order):
```bash
uv run train-dcgan
uv run train-vqvae
uv run train-ddpm
```

Each script is config-driven; override hyperparameters from the CLI:
```bash
uv run train-dcgan --z_dim 64 --lr_g 1e-4
uv run train-ddpm  --timesteps 500 --base_channels 64
```

Checkpoints are saved to `checkpoints/<model>/` and sample grids to `outputs/<model>/` every 10 epochs.

**Run full evaluation** (FID + interpolation for all models):
```bash
uv run python evaluate_all.py
```

**Run ablation studies**:
```bash
uv run python ablate_dcgan.py
uv run python ablate_ddpm_steps.py
uv run python evaluate_cats_and_dogs.py
```

**Open notebooks**:
```bash
uv run jupyter notebook
```

## Results

| Model | FID ↓ | Notes |
|---|---|---|
| DCGAN (WGAN-GP) | 221.51 | z_dim=128, 100 epochs |
| VQ-VAE + PixelCNN | 187.54 | K=512, D=64, 100+100 epochs |
| **DDPM (DDIM 50 steps)** | **24.36** | T=1000, 200 epochs, EMA |

DDPM achieves substantially lower FID, which is consistent with results reported in the diffusion models literature for similar-scale experiments. Key ablation findings (DCGAN ablations are 3-seed mean ± std):
- **DDIM steps** (DDPM): FID drops from 66.92 (10 steps) to 39.81 (200 steps); 50 steps (FID 54.51 without EMA, 24.36 with EMA) is the practical trade-off.
- **Latent dim** (DCGAN): z_dim=64 gives best mean FID (246.65 ± 5.57) vs baseline z_dim=128 (264.25 ± 20.21); z_dim=256 is equivalent to baseline. Baseline has high seed variance.
- **Learning rate** (DCGAN): lr=4e-4 gives best mean FID (220.05 ± 4.47, −44 pts vs baseline); lr=1e-4 is worst (295.11 ± 18.27).
- **Cats+Dogs dataset**: mixed DCGAN achieves FID 206.57 vs cats and 197.79 vs dogs, compared to the cats-only baseline of 221.51 (improvement is within single-seed variance; see report for caveats).

## Experiments
We systematically investigate the following factors:
1. **Latent Dimensionality:** Impact of the latent noise vector z-dimension on generation quality.
2. **Learning Rate:** Impact of learning rate on DCGAN training stability and FID.
3. **Stability:** Techniques to prevent mode collapse: label smoothing (DCGAN), WGAN-GP gradient penalty, EMA weights (DDPM).
4. **Dataset Complexity:** Comparing models trained on cats only versus a mixed cats and dogs dataset.

## Evaluation Metrics
Model performance is assessed through:
* **Quantitative:** Fréchet Inception Distance (FID) via `torch-fidelity`, computed on 5,000 generated vs. 5,000 real images.
* **Qualitative:** Visual assessment of realism (e.g., anatomical correctness of ears and whiskers) and sample diversity.
* **Latent Space Interpolation:** A 10-image sequence generated by linear/spherical interpolation between two latent vectors to evaluate latent space continuity.

## Implementation Notes
* All experiments use `SEED=42` (set globally via `src/utils/seed.py`) for reproducibility.
* Windows users: use `num_workers=0` in DataLoaders when running in interactive/notebook sessions.
* YAML configs must be opened with UTF-8 encoding; this is handled automatically by the training scripts.
