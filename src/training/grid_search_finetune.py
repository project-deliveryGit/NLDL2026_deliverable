import os
import csv
import sys
import yaml
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset import PhantomTripletDataset, make_case_balanced_sampler, split_dataset
from utils.metrics import dice_coef_from_probs
from utils.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def flat_config(cfg, grid_values):
    flat = {}
    for section in ("data", "model", "training"):
        flat.update(cfg[section])
    flat.update(grid_values)
    flat["use_augmentation"] = True
    return flat


def get_run_name(grid_values):
    parts = [f"{k}={v}" for k, v in grid_values.items()]
    return "_".join(parts)


def get_completed_runs(results_csv):
    if not os.path.exists(results_csv):
        return set()
    with open(results_csv) as f:
        return {row["run_name"] for row in csv.DictReader(f) if row.get("run_name")}


def train(config, pretrained_path, run_dir, device):
    os.makedirs(run_dir, exist_ok=True)

    all_cases = sorted(
        d for d in os.listdir(config["phantom_root"])
        if os.path.isdir(os.path.join(config["phantom_root"], d)) and d.startswith("US-Acq_")
    )
    test_case = next(c for c in all_cases if c.startswith(config["test_case"]))
    train_val_cases = [c for c in all_cases if c != test_case]

    train_ds = PhantomTripletDataset(config["phantom_root"], train_val_cases, config)
    test_ds  = PhantomTripletDataset(config["phantom_root"], [test_case], {**config, "use_augmentation": False})

    train_set, val_set = split_dataset(train_ds, config["val_split"], config["split_seed"])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              sampler=make_case_balanced_sampler(train_set),
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True)

    model = build_model(config, pretrained_path=pretrained_path, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                    factor=0.5, patience=config.get("scheduler_patience", 5))
    scaler    = GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()

    best_val, no_improve, best_path = -1.0, 0, os.path.join(run_dir, "best.pth")

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        by_case = {}
        with torch.no_grad():
            for imgs, masks, case_ids in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                probs = torch.sigmoid(model(imgs))
                for cid, d in zip(case_ids, dice_coef_from_probs(probs, masks).cpu().numpy()):
                    by_case.setdefault(cid, []).append(float(d))

        macro_dice = float(np.mean([np.mean(v) for v in by_case.values()])) if by_case else 0.0
        scheduler.step(macro_dice)

        tag = "NEW BEST" if macro_dice > best_val else f"no improve {no_improve}/{config['patience']}"
        print(f"  [{epoch:02d}/{config['max_epochs']}] loss={train_loss:.4f}  val={macro_dice:.4f}  {tag}")

        if macro_dice > best_val:
            best_val, no_improve = macro_dice, 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if no_improve >= config["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    test_dices = []
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))
            test_dices.extend(dice_coef_from_probs(probs, masks).cpu().numpy())

    test_dices = np.array(test_dices)
    return {
        "val_mean":  float(best_val),
        "test_mean": float(np.mean(test_dices)),
        "test_std":  float(np.std(test_dices, ddof=1)) if len(test_dices) > 1 else 0.0,
        "n_frames":  len(test_dices),
    }


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    grid        = cfg["grid"]
    keys        = list(grid.keys())
    combos      = list(itertools.product(*[grid[k] for k in keys]))
    results_csv = os.path.join(args.output_dir, "results.csv")
    completed   = get_completed_runs(results_csv)

    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            csv.writer(f).writerow(keys + ["val_mean", "test_mean", "test_std", "n_frames", "run_name"])

    print(f"\nGrid search: {len(combos)} runs  |  completed: {len(completed)}\n")

    for i, combo in enumerate(combos, 1):
        grid_values = dict(zip(keys, combo))
        run_name    = get_run_name(grid_values)

        if run_name in completed:
            print(f"[{i}/{len(combos)}] SKIP  {run_name}")
            continue

        print(f"\n[{i}/{len(combos)}] {run_name}")
        config  = flat_config(cfg, grid_values)
        run_dir = os.path.join(args.output_dir, f"run_{i:03d}_{run_name}")

        try:
            result = train(config, args.pretrained, run_dir, device)
            with open(results_csv, "a", newline="") as f:
                csv.writer(f).writerow(list(combo) + [
                    result["val_mean"], result["test_mean"],
                    result["test_std"], result["n_frames"], run_name
                ])
            print(f"  val={result['val_mean']:.4f}  test={result['test_mean']:.4f} ± {result['test_std']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()