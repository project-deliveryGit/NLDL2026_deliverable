import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.models import load_standard_model
from utils.training import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True)
    parser.add_argument("--baseline",     required=True)
    parser.add_argument("--patient_img",  required=True)
    parser.add_argument("--patient_mask", required=True)
    parser.add_argument("--phantom_img",  required=True)
    parser.add_argument("--phantom_mask", required=True)
    parser.add_argument("--output",       required=True)
    return parser.parse_args()


def load_image(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def load_mask(path, size=512):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.float32) / 255.0


def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def to_tensor(img, device):
    triplet = np.stack([img, img, img], axis=-1)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(Image.fromarray(triplet))
    return t.unsqueeze(0).to(device)


def predict(model, img_tensor):
    with torch.no_grad():
        return torch.sigmoid(model(img_tensor)).cpu().squeeze().numpy()


def dice(pred, gt, eps=1e-6):
    p, g = (pred > 0.5).astype(np.float32), (gt > 0.5).astype(np.float32)
    return (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def make_overlay(img_show, pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    overlay = np.zeros((*gt_b.shape, 3))
    overlay[gt_b & pred_b]  = [0, 1, 0]
    overlay[~gt_b & pred_b] = [1, 0, 0]
    overlay[gt_b & ~pred_b] = [0, 0, 1]
    bg  = np.stack([img_show] * 3, axis=-1)
    return np.where(overlay.sum(axis=-1, keepdims=True) > 0, 0.5 * overlay + 0.5 * bg, bg)


def crop_and_stretch(img, pred, mask, zoom=1.2):
    row_means = img.mean(axis=1)
    non_black = np.where(row_means > 0.04)[0]
    if len(non_black) == 0:
        return img, pred, mask
    top, bottom = non_black[0], non_black[-1] + 1
    img, pred, mask = img[top:bottom], pred[top:bottom], mask[top:bottom]
    h, w = img.shape
    new_h = int(h * zoom)
    img  = cv2.resize(img,  (w, new_h), interpolation=cv2.INTER_LINEAR)
    pred = cv2.resize(pred.astype(np.float32), (w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.float32), (w, new_h), interpolation=cv2.INTER_NEAREST)
    return img, pred, mask


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = {}
    for section in ("data", "model", "training"):
        base_config.update(cfg[section])
    base_config["encoder_weights"] = None

    model = load_standard_model(args.baseline, base_config, device)
    model.eval()

    patient_img  = load_image(args.patient_img)
    patient_mask = load_mask(args.patient_mask)
    phantom_img  = load_image(args.phantom_img)
    phantom_mask = load_mask(args.phantom_mask)

    patient_pred = predict(model, to_tensor(patient_img, device))
    phantom_pred = predict(model, to_tensor(phantom_img, device))

    patient_dice = dice(patient_pred, patient_mask)
    phantom_dice = dice(phantom_pred, phantom_mask)
    print(f"Patient Dice: {patient_dice:.4f}")
    print(f"Phantom Dice: {phantom_dice:.4f}")

    patient_show = normalize(patient_img)
    phantom_show = normalize(phantom_img)
    patient_show, patient_pred, patient_mask = crop_and_stretch(patient_show, patient_pred, patient_mask)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].set_title("Patient Frame",                              fontsize=20, fontweight="bold")
    axes[0, 1].set_title(f"Baseline Prediction\nDice={patient_dice:.3f}", fontsize=20, fontweight="bold")
    axes[1, 0].set_title("Phantom Frame",                              fontsize=20, fontweight="bold")
    axes[1, 1].set_title(f"Baseline Prediction\nDice={phantom_dice:.3f}", fontsize=20, fontweight="bold")

    axes[0, 0].imshow(patient_show, cmap="gray")
    axes[0, 1].imshow(make_overlay(patient_show, patient_pred, patient_mask))
    axes[1, 0].imshow(phantom_show, cmap="gray")
    axes[1, 1].imshow(make_overlay(phantom_show, phantom_pred, phantom_mask))

    for ax in axes.flat:
        ax.axis("off")

    legend = [
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red",   label="FP"),
        mpatches.Patch(color="blue",  label="FN"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=16, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1.0])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()