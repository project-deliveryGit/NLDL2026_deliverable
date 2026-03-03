import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset import PhantomTripletDataset
from utils.metrics import compute_metrics
from utils.models import load_standard_model, load_lora_model, load_cross_attention_model
from utils.training import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True,  help="Path to any base config yaml")
    parser.add_argument("--phantom_root",   required=True)
    parser.add_argument("--test_case",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--baseline_path",  required=True)
    parser.add_argument("--finetune_path",  required=True)
    parser.add_argument("--lora_path",      required=True)
    parser.add_argument("--lora_r",         type=int, required=True)
    parser.add_argument("--lora_alpha",     type=int, required=True)
    parser.add_argument("--cross_attn_path",     required=True)
    parser.add_argument("--cross_attn_layers",   type=int, nargs="+", required=True)
    parser.add_argument("--cross_attn_heads",    type=int, default=2)
    parser.add_argument("--cross_attn_downsample", type=int, default=4)
    return parser.parse_args()


def get_model_stats(model, device, n_warmup=10, n_runs=50):
    dummy = torch.randn(1, 3, 512, 512).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb     = (param_size + buffer_size) / 1024 / 1024

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    avg_ms = np.mean(times)
    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "size_mb":          size_mb,
        "inference_ms":     avg_ms,
        "fps":              1000.0 / avg_ms,
    }


def evaluate(model, loader, device):
    model.eval()
    all_dice, all_iou, all_recall, all_precision = [], [], [], []
    frame_results = {}

    with torch.no_grad():
        for imgs, masks, indices in tqdm(loader, ncols=80):
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))
            for i, idx in enumerate(indices):
                d, iou, r, p = compute_metrics(probs[i], masks[i])
                all_dice.append(d)
                all_iou.append(iou)
                all_recall.append(r)
                all_precision.append(p)
                frame_results[idx.item()] = {
                    "prob": probs[i].cpu().squeeze().numpy(),
                    "mask": masks[i].cpu().squeeze().numpy(),
                    "img":  imgs[i].cpu().squeeze().numpy(),
                    "dice": d,
                }

    stats = get_model_stats(model, device)

    return {
        "dice_mean":      np.mean(all_dice),
        "dice_std":       np.std(all_dice, ddof=1),
        "iou_mean":       np.mean(all_iou),
        "iou_std":        np.std(all_iou, ddof=1),
        "recall_mean":    np.mean(all_recall),
        "recall_std":     np.std(all_recall, ddof=1),
        "precision_mean": np.mean(all_precision),
        "precision_std":  np.std(all_precision, ddof=1),
        **stats,
    }, frame_results


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    base_config = {}
    for section in ("data", "model", "training"):
        base_config.update(cfg[section])
    base_config["phantom_root"]  = args.phantom_root
    base_config["use_augmentation"] = False

    dataset = PhantomTripletDataset(args.phantom_root, [args.test_case], base_config)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test frames: {len(dataset)}\n")

    lora_config = {**base_config, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    cross_config = {
        **base_config,
        "attention_layers":    args.cross_attn_layers,
        "attention_heads":     args.cross_attn_heads,
        "attention_downsample": args.cross_attn_downsample,
    }

    models = {
        "Baseline":       lambda: load_standard_model(args.baseline_path,   base_config, device),
        "Fine-tuning":    lambda: load_standard_model(args.finetune_path,   base_config, device),
        "LoRA":           lambda: load_lora_model(args.lora_path,           lora_config, device),
        "Cross-Attention":lambda: load_cross_attention_model(
                                    args.cross_attn_path, cross_config, args.baseline_path, device),
    }

    all_results      = {}
    all_frame_results = {}

    for name, build in models.items():
        print(f"{'='*50}\nEvaluating: {name}\n{'='*50}")
        model = build()
        metrics, frame_results = evaluate(model, loader, device)
        all_results[name]       = metrics
        all_frame_results[name] = frame_results

        print(f"  Dice:      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
        print(f"  IoU:       {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"  Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
        print(f"  Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
        print(f"  FPS:       {metrics['fps']:.1f}  |  Size: {metrics['size_mb']:.1f} MB\n")

        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items() if not isinstance(mv, np.ndarray)}
                   for k, v in all_results.items()}, f, indent=4)

    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'FPS':>8} {'MB':>8}")
    print(f"{'='*70}")
    for name, m in all_results.items():
        print(f"{name:<20} {m['dice_mean']:.4f} {m['iou_mean']:.4f} "
              f"{m['recall_mean']:.4f} {m['precision_mean']:.4f} "
              f"{m['fps']:>8.1f} {m['size_mb']:>8.1f}")

    return all_frame_results


if __name__ == "__main__":
    main()