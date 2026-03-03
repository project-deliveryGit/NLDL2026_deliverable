import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.models import build_model, CrossAttentionUNet
from utils.training import (load_config, flat_config, get_completed_runs,
                             write_csv_header, save_result_row, setup_loaders, train_loop)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--pretrained",  required=True)
    parser.add_argument("--output_dir",  required=True)
    return parser.parse_args()


def get_run_name(attention_layers):
    return "attn_layers_" + "_".join(map(str, attention_layers))


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    layer_combos = cfg["grid"]["attention_layers"]
    results_csv  = os.path.join(args.output_dir, "results.csv")
    completed    = get_completed_runs(results_csv)

    if not os.path.exists(results_csv):
        write_csv_header(results_csv, ["attention_layers"])

    print(f"\nCross-attention grid search: {len(layer_combos)} runs  |  completed: {len(completed)}\n")

    for i, attention_layers in enumerate(layer_combos, 1):
        run_name = get_run_name(attention_layers)

        if run_name in completed:
            print(f"[{i}/{len(layer_combos)}] SKIP  {run_name}")
            continue

        print(f"\n[{i}/{len(layer_combos)}] {run_name}")
        config  = flat_config(cfg, {"attention_layers": attention_layers})
        run_dir = os.path.join(args.output_dir, f"run_{i:02d}_{run_name}")

        try:
            train_loader, val_loader, test_loader = setup_loaders(config)
            patient_model = build_model({**config, "encoder_weights": None},
                                        pretrained_path=args.pretrained, device=device)
            model  = CrossAttentionUNet(patient_model, config, device=device).to(device)
            result = train_loop(model, config, train_loader, val_loader, test_loader, run_dir, device)
            save_result_row(results_csv, [str(attention_layers)], result, run_name)
            print(f"  val={result['val_mean']:.4f}  test={result['test_mean']:.4f} ± {result['test_std']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()