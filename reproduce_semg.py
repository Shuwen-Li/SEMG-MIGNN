#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from script.MIGNN_model import Graph_DataLoader, MIGNN_model1, MIGNN_model2


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "reproduction_data"
BEST_CONFIG = {
    "data1": {"seed": 7, "batch_size": 32, "learning_rate": 1e-3,
              "hidden_size": 128, "linear_depth": 4, "patience": 15},
    "data2": {"seed": 77, "batch_size": 32, "learning_rate": 5e-4,
              "hidden_size": 384, "linear_depth": 2, "fc_size": 256,
              "patience": 15},
}


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred
    return {
        "n": len(y_true),
        "r2": float(1 - np.sum(residual**2) / np.sum((y_true - y_true.mean()) ** 2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def score_official(output_dir):
    rows = []
    for truth_path in sorted((ROOT / "result").glob("*_y_val*.npy")):
        pred_path = Path(str(truth_path).replace("_y_val", "_y_pred"))
        if pred_path.exists():
            row = {"experiment": truth_path.stem.replace("_y_val", "")}
            row.update(metrics(np.load(truth_path), np.load(pred_path)))
            rows.append(row)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "official_metrics.csv", index=False)
    print(frame.to_string(index=False))


def generate(dataset, output_dir):
    """Build the dense steric/electronic tensors used by the training notebooks."""
    from script.SEMG import Calc_SPMS, PackCub

    base = ROOT / "Data" / dataset
    steric_dir = base / "xtb_sdf"
    electronic_dir = base / "b3lyp_def2svp"
    frame = pd.read_csv(base / f"{dataset}.csv")
    if dataset == "data1":
        columns = ("Ligand", "Additive", "Base", "Aryl halide")
        prefixes = ("lig", "add", "base", "ar_ha")
        radius = None
    else:
        columns = ("Catalyst", "Imine", "Thiol")
        prefixes = ("cat", "imine", "thiol")
        radius = 7
    with (steric_dir / "smiles_file_dict.csv").open() as handle:
        smiles_to_name = dict(line.strip().split(",", 1) for line in handle if line.strip())
    spms_parts, electronic_parts = [], []
    for column, prefix in zip(columns, prefixes):
        files = sorted(
            steric_dir.glob(f"{prefix}_*.sdf"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        kwargs = {} if radius is None else {"sphere_radius": radius}
        calculator = Calc_SPMS([str(path) for path in files], str(steric_dir) + "/", **kwargs)
        spms, name_to_index = calculator.calc_spms([str(path) for path in files], str(steric_dir) + "/")
        names = [smiles_to_name[smiles] for smiles in frame[column].to_numpy()]
        component_spms = np.asarray([spms[name_to_index[name]] for name in names])
        component_electronic = PackCub(np.asarray([
            np.load(electronic_dir / f"def2-svp_b3lyp_{name}.npy") for name in names
        ]))
        spms_parts.append(np.transpose(component_spms, (0, 2, 3, 1)))
        electronic_parts.append(np.transpose(component_electronic, (0, 2, 3, 4, 1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{dataset}_react_spms.npy", np.concatenate(spms_parts, axis=3))
    np.save(output_dir / f"{dataset}_react_elec.npy", np.concatenate(electronic_parts, axis=4))
    np.save(output_dir / f"{dataset}_labels.npy", frame["Output"].to_numpy())


def make_model(dataset):
    config = BEST_CONFIG[dataset]
    if dataset == "data1":
        return MIGNN_model1(
            lig_size=107, add_size=36, base_size=56, ar_ha_size=18,
            hidden_size=config["hidden_size"], linear_depth=config["linear_depth"], atom_attention=1,
            inter_attention=1, end_attention=1, spms_number=32,
            ele_number=32, inter_len=8, final_act="none",
        )
    return MIGNN_model2(
        cat_size=184, imine_size=33, thiol_size=19, hidden_size=config["hidden_size"],
        linear_depth=config["linear_depth"], atom_attention=1, inter_attention=1, end_attention=1,
        inter_len=8, spms_number=32, ele_number=32, fc_size=config["fc_size"],
        final_act="dense",
    )


def train(dataset, input_dir, output_dir, model_dir, epochs, seed=None):
    config = BEST_CONFIG[dataset]
    seed = config["seed"] if seed is None else seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    spms = np.load(input_dir / f"{dataset}_react_spms.npy", mmap_mode="r")
    elec = np.load(input_dir / f"{dataset}_react_elec.npy", mmap_mode="r")
    labels = np.load(input_dir / f"{dataset}_labels.npy")
    order = np.arange(len(labels))
    np.random.shuffle(order)
    split = int(0.7 * len(labels)) if dataset == "data1" else 600
    batch_size = config["batch_size"]
    train_idx, val_idx = order[:split], order[split:]
    y_train, y_val = labels[train_idx], labels[val_idx]
    scale = y_train.max() - y_train.min()
    train_loader = Graph_DataLoader(
        spms[train_idx], elec[train_idx], (y_train - y_train.min()) / scale,
        batch_size=batch_size,
    )
    val_loader = Graph_DataLoader(
        spms[val_idx], elec[val_idx], (y_val - y_train.min()) / scale,
        batch_size=batch_size,
    )
    pred_loader = Graph_DataLoader(
        spms[val_idx], elec[val_idx], (y_val - y_train.min()) / scale,
        batch_size=batch_size, predict=True,
    )
    model = make_model(dataset)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="mse")
    started = time.time()
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["patience"], restore_best_weights=True,
    )]
    history = model.fit(
        train_loader, validation_data=val_loader, epochs=epochs,
        callbacks=callbacks, verbose=2,
    )
    prediction = model.predict(pred_loader, verbose=0).reshape(-1) * scale + y_train.min()
    if dataset == "data1":
        prediction[prediction < 0] = 0
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / dataset
    model.save_weights(str(model_path))
    report = {
        "dataset": dataset,
        "max_epochs": epochs,
        "epochs_ran": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": config["learning_rate"],
        "hidden_size": config["hidden_size"],
        "linear_depth": config["linear_depth"],
        "weights_path": str(model_path),
        "seconds": time.time() - started,
        **metrics(y_val, prediction),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{dataset}_y_val.npy", y_val)
    np.save(output_dir / f"{dataset}_y_pred.npy", prediction)
    pd.DataFrame(history.history).to_csv(output_dir / f"{dataset}_history.csv", index=False)
    (output_dir / f"{dataset}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (model_dir / f"{dataset}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("official", "generate1", "generate2", "data1", "data2"))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction_results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "trained_model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int,
                        help="override the tuned dataset-specific default seed")
    args = parser.parse_args()
    if args.command == "official":
        score_official(args.output_dir)
    elif args.command.startswith("generate"):
        generate("data" + args.command[-1], args.input_dir)
    else:
        default_epochs = 100 if args.command == "data1" else 50
        train(args.command, args.input_dir, args.output_dir, args.model_dir,
              args.epochs or default_epochs, args.seed)


if __name__ == "__main__":
    main()
