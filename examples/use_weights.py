"""Load a standard mimiLLM model directory and generate a continuation."""

import argparse
import sys
from pathlib import Path

from mimillm import generate_text, load_model


parser = argparse.ArgumentParser()
parser.add_argument("weights", type=Path, help="Directory with config.json and model.safetensors")
parser.add_argument("--prompt", default="Once upon a time")
args = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

model = load_model(args.weights)
text = generate_text(model, args.prompt, max_new_tokens=40, temperature=0.7, top_k=20)
print(args.prompt + text)
