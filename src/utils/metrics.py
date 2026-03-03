def dice_coef_from_probs(probs, target, eps=1e-6):
    p = (probs > 0.5).float()
    num = (p * target).sum((1, 2, 3)) * 2 + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return num / den


def compute_metrics(probs, target, threshold=0.5, eps=1e-6):
    pred = (probs > threshold).float()
    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()
    dice      = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou       = (TP + eps) / (TP + FP + FN + eps)
    recall    = (TP + eps) / (TP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    return dice, iou, recall, precision