import os
import sys
import argparse
import itertools
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.models import build_model
from utils.training import (load_config, flat_config, get_run_name, get_completed_runs,
                             write_csv_header, save_result_row, setup_loaders, train_loop)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--pretrained",  required=True)
    parser.add_argument("--output_dir",  required=True)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    grid        = cfg["grid"]
    keys        = list(grid.keys())
    combos      = list(itertools.product(*[grid[k] for k in keys]))
    results_csv = os.path.join(args.output_dir, "results.csv")
    completed   = get_completed_runs(results_csv)

    if not os.path.exists(results_csv):
        write_csv_header(results_csv, keys)

    print(f"\nFine-tune grid search: {len(combos)} runs  |  completed: {len(completed)}\n")

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
            train_loader, val_loader, test_loader = setup_loaders(config)
            model  = build_model(config, pretrained_path=args.pretrained, device=device)
            result = train_loop(model, config, train_loader, val_loader, test_loader, run_dir, device)
            save_result_row(results_csv, combo, result, run_name)
            print(f"  val={result['val_mean']:.4f}  test={result['test_mean']:.4f} ± {result['test_std']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()