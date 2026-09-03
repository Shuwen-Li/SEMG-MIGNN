#!/usr/bin/env python3
"""Successive-halving style hyperparameter search for the SEMG-MIGNN models."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from reproduce_semg import DEFAULT_INPUT, ROOT, metrics
from script.MIGNN_model import Graph_DataLoader, MIGNN_model1, MIGNN_model2


SEARCH_SPACE = {
    "data1": [
        {"name": "paper", "lr": 1e-3, "batch": 128, "hidden": 128, "depth": 4},
        {"name": "lr_3e-4", "lr": 3e-4, "batch": 128, "hidden": 128, "depth": 4},
        {"name": "lr_5e-4", "lr": 5e-4, "batch": 128, "hidden": 128, "depth": 4},
        {"name": "lr_2e-3", "lr": 2e-3, "batch": 128, "hidden": 128, "depth": 4},
        {"name": "batch64", "lr": 1e-3, "batch": 64, "hidden": 128, "depth": 4},
        {"name": "hidden64_depth3", "lr": 1e-3, "batch": 128, "hidden": 64, "depth": 3},
        {"name": "hidden256_depth3", "lr": 1e-3, "batch": 128, "hidden": 256, "depth": 3},
    ],
    "data2": [
        {"name": "paper", "lr": 1e-3, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_3e-4", "lr": 3e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_5e-4", "lr": 5e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_2e-3", "lr": 2e-3, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "batch64", "lr": 1e-3, "batch": 64, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "hidden256", "lr": 1e-3, "batch": 32, "hidden": 256, "depth": 2, "fc": 256},
        {"name": "hidden256_depth3", "lr": 1e-3, "batch": 32, "hidden": 256, "depth": 3, "fc": 256},
    ],
}

REFINED_SEARCH_SPACE = {
    "data1": [
        {"name": "best_round1", "lr": 1e-3, "batch": 64, "hidden": 128, "depth": 4},
        {"name": "lr_7e-4", "lr": 7e-4, "batch": 64, "hidden": 128, "depth": 4},
        {"name": "lr_1p5e-3", "lr": 1.5e-3, "batch": 64, "hidden": 128, "depth": 4},
        {"name": "batch32", "lr": 1e-3, "batch": 32, "hidden": 128, "depth": 4},
        {"name": "depth3", "lr": 1e-3, "batch": 64, "hidden": 128, "depth": 3},
        {"name": "hidden256", "lr": 1e-3, "batch": 64, "hidden": 256, "depth": 4},
    ],
    "data2": [
        {"name": "best_round1", "lr": 5e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_4e-4", "lr": 4e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_6e-4", "lr": 6e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "lr_7p5e-4", "lr": 7.5e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "batch16", "lr": 5e-4, "batch": 16, "hidden": 512, "depth": 2, "fc": 256},
        {"name": "hidden384", "lr": 5e-4, "batch": 32, "hidden": 384, "depth": 2, "fc": 256},
        {"name": "depth3", "lr": 5e-4, "batch": 32, "hidden": 512, "depth": 3, "fc": 256},
        {"name": "fc128", "lr": 5e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 128},
        {"name": "fc512", "lr": 5e-4, "batch": 32, "hidden": 512, "depth": 2, "fc": 512},
    ],
}


def model_for(dataset, config):
    if dataset == "data1":
        return MIGNN_model1(
            lig_size=107, add_size=36, base_size=56, ar_ha_size=18,
            hidden_size=config["hidden"], linear_depth=config["depth"],
            atom_attention=1, inter_attention=1, end_attention=1,
            spms_number=32, ele_number=32, inter_len=8, final_act="none",
        )
    return MIGNN_model2(
        cat_size=184, imine_size=33, thiol_size=19,
        hidden_size=config["hidden"], linear_depth=config["depth"],
        atom_attention=1, inter_attention=1, end_attention=1, inter_len=8,
        spms_number=32, ele_number=32, fc_size=config["fc"], final_act="dense",
    )


def load_split(dataset, input_dir, seed):
    spms = np.load(input_dir / f"{dataset}_react_spms.npy", mmap_mode="r")
    elec = np.load(input_dir / f"{dataset}_react_elec.npy", mmap_mode="r")
    labels = np.load(input_dir / f"{dataset}_labels.npy")
    rng = np.random.RandomState(seed)
    order = np.arange(len(labels))
    rng.shuffle(order)
    split = int(0.7 * len(labels)) if dataset == "data1" else 600
    return spms, elec, labels, order[:split], order[split:]


def run_trial(dataset, arrays, config, epochs, patience, output_dir, stage, seed):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    spms, elec, labels, train_idx, val_idx = arrays
    y_train, y_val = labels[train_idx], labels[val_idx]
    y_min, scale = y_train.min(), y_train.max() - y_train.min()
    train_loader = Graph_DataLoader(
        spms[train_idx], elec[train_idx], (y_train - y_min) / scale,
        batch_size=config["batch"],
    )
    val_loader = Graph_DataLoader(
        spms[val_idx], elec[val_idx], (y_val - y_min) / scale,
        batch_size=config["batch"],
    )
    pred_loader = Graph_DataLoader(
        spms[val_idx], elec[val_idx], (y_val - y_min) / scale,
        batch_size=config["batch"], predict=True,
    )
    model = model_for(dataset, config)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["lr"]), loss="mse")
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )]
    started = time.time()
    history = model.fit(
        train_loader, validation_data=val_loader, epochs=epochs,
        callbacks=callbacks, verbose=0,
    )
    prediction = model.predict(pred_loader, verbose=0).reshape(-1) * scale + y_min
    if dataset == "data1":
        prediction[prediction < 0] = 0
    report = {
        "stage": stage, "seed": seed, **config,
        "epochs_ran": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "seconds": time.time() - started,
        **metrics(y_val, prediction),
    }
    prefix = output_dir / f"{dataset}_{stage}_{config['name']}"
    pd.DataFrame(history.history).to_csv(str(prefix) + "_history.csv", index=False)
    np.save(str(prefix) + "_y_val.npy", y_val)
    np.save(str(prefix) + "_y_pred.npy", prediction)
    print(json.dumps(report), flush=True)
    del model, train_loader, val_loader, pred_loader
    gc.collect()
    return report


def tune(dataset, input_dir, output_dir, shortlist, phase, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_split(dataset, input_dir, seed)
    search_space = SEARCH_SPACE[dataset] if phase == "coarse" else REFINED_SEARCH_SPACE[dataset]
    screening_epochs = (25 if dataset == "data1" else 20) if phase == "coarse" else (35 if dataset == "data1" else 25)
    screening = [run_trial(
        dataset, arrays, config, screening_epochs, 6, output_dir, "screen", seed,
    ) for config in search_space]
    finalists = sorted(screening, key=lambda row: row["r2"], reverse=True)[:shortlist]
    by_name = {config["name"]: config for config in search_space}
    full_epochs = 100 if dataset == "data1" else 50
    final = [run_trial(
        dataset, arrays, by_name[row["name"]], full_epochs, 15,
        output_dir, "final", seed,
    ) for row in finalists]
    all_rows = screening + final
    pd.DataFrame(all_rows).to_csv(output_dir / f"{dataset}_tuning_summary.csv", index=False)
    winner = max(final, key=lambda row: row["r2"])
    (output_dir / f"{dataset}_best.json").write_text(json.dumps(winner, indent=2) + "\n")
    print(f"BEST {dataset}: {json.dumps(winner)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("data1", "data2"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tuning_results")
    parser.add_argument("--shortlist", type=int, default=2)
    parser.add_argument("--phase", choices=("coarse", "refined"), default="coarse")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--require-gpu", action="store_true",
                        help="abort instead of silently falling back to CPU")
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU is available. Check /dev/nvidia-uvm and the NVIDIA driver."
        )
    print(f"TensorFlow {tf.__version__}; GPUs: {gpus}", flush=True)
    tune(args.dataset, args.input_dir, args.output_dir, args.shortlist, args.phase, args.seed)


if __name__ == "__main__":
    main()
