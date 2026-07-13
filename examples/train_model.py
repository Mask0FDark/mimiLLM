"""Train a model from this project's config and four data directories."""

from pathlib import Path

from mimillm import train_from_config


PROJECT_DIR = Path(__file__).resolve().parents[1]

result = train_from_config(
    PROJECT_DIR / "configs" / "debug.json",
    output_dir=PROJECT_DIR / "weights",
)

print(f"Reusable weights: {result.weights_dir}")
print(f"Resume checkpoint: {result.checkpoint_path}")
