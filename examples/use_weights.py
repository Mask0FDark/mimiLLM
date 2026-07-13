"""Load standard mimiLLM weights and use one mixed language model."""

import argparse
import sys
from pathlib import Path

from mimillm import generate_response, load_model


parser = argparse.ArgumentParser()
parser.add_argument("weights", type=Path, help="Directory with config.json and model.safetensors")
parser.add_argument("--prompt", default="What is a token?")
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", type=int, default=1)
args = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

model = load_model(args.weights)
print(generate_response(
    model,
    args.prompt,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
))
