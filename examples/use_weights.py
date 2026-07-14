"""Load standard mimiLLM weights and use one mixed language model."""

import argparse
import sys
from pathlib import Path

from mimillm import generate_response, load_model


parser = argparse.ArgumentParser(description="Run a language model saved by mimiLLM")
parser.add_argument("weights", type=Path, help="directory containing config.json and model.safetensors")
parser.add_argument("--prompt", default="What is a token?", help="question or instruction for the model")
parser.add_argument(
    "--max-new-tokens", type=int, default=256,
    help="maximum number of byte tokens to generate (default: 256)",
)
parser.add_argument(
    "--temperature", type=float, default=0.0,
    help="0 selects the most likely token; values above 0 enable sampling (default: 0)",
)
parser.add_argument(
    "--top-k", type=int, default=20,
    help="number of most likely candidate tokens used during sampling (default: 20)",
)
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
