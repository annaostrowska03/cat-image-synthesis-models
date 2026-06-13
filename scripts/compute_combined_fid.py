"""Compute FID of cats+dogs DCGAN against combined (cat+dog) reference set.

Uses already-saved PNG images - no retraining or generation needed.
Reference: 2500 real cats + 2500 real dogs = 5000 combined images.
Generated: 5000 images from cats_and_dogs DCGAN (already in fid_samples/).
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.fid import compute_fid

REAL_CATS = Path("outputs/real_64")
REAL_DOGS = Path("outputs/real_dogs_64")
FAKE_DIR = Path("outputs/cats_and_dogs/fid_samples")
COMBINED = Path("outputs/real_cats_dogs_64")
RESULTS = Path("outputs/cats_and_dogs/fid_results.json")
N_EACH = 2500


def build_combined_ref() -> None:
    if COMBINED.exists() and len(list(COMBINED.glob("*.png"))) >= N_EACH * 2:
        print(f"Combined reference already exists ({COMBINED})")
        return
    COMBINED.mkdir(parents=True, exist_ok=True)
    cats = sorted(REAL_CATS.glob("*.png"))[:N_EACH]
    dogs = sorted(REAL_DOGS.glob("*.png"))[:N_EACH]
    for i, src in enumerate(cats):
        shutil.copy(src, COMBINED / f"cat_{i:05d}.png")
    for i, src in enumerate(dogs):
        shutil.copy(src, COMBINED / f"dog_{i:05d}.png")
    print(f"Built combined reference: {len(cats)} cats + {len(dogs)} dogs -> {COMBINED}")


def main() -> None:
    build_combined_ref()

    print("Computing FID vs combined (cats+dogs) reference ...")
    fid_combined = compute_fid(str(COMBINED), str(FAKE_DIR))
    print(f"  FID vs cats+dogs: {fid_combined:.2f}")

    results = {}
    if RESULTS.exists():
        with open(RESULTS) as f:
            results = json.load(f)
    results["fid_vs_combined"] = fid_combined
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated {RESULTS}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
