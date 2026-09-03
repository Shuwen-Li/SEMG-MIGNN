#!/usr/bin/env python3
"""Run one full SEMG seed-sweep trial with a fixed tuned architecture."""

import argparse
import json
from pathlib import Path

from tune_semg import load_split, run_trial


CONFIGS = {
    "data1": {"name": "seed_sweep", "lr": 1e-3, "batch": 32, "hidden": 128, "depth": 4},
    "data2": {"name": "seed_sweep", "lr": 5e-4, "batch": 32, "hidden": 384, "depth": 2, "fc": 256},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("data1", "data2"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--tf-seed", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    tf_seed = args.seed if args.tf_seed is None else args.tf_seed
    arrays = load_split(args.dataset, args.input_dir, split_seed)
    config = dict(CONFIGS[args.dataset])
    if args.batch is not None:
        config["batch"] = args.batch
    if args.lr is not None:
        config["lr"] = args.lr
    epochs = 100 if args.dataset == "data1" else 50
    result = run_trial(
        args.dataset, arrays, config, epochs, 15,
        args.output_dir, f"split{split_seed}_tf{tf_seed}", tf_seed,
    )
    result["split_seed"] = split_seed
    result["tensorflow_seed"] = tf_seed
    (args.output_dir / f"{args.dataset}_seed{args.seed}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
