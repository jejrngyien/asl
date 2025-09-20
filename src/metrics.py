"""
Basic metrics: Top-k accuracy and a running confusion matrix.
"""
from typing import Iterable
import torch
import torch.nn as nn


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Compute the top-k accuracies for the specified values of k.
    """
    maxk = max(topk)
    with torch.no_grad():
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / target.size(0))).item())
        return res


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds/targets: 1D tensors of ints
        for t, p in zip(targets.view(-1), preds.view(-1)):
            self.mat[t.long(), p.long()] += 1

    def compute(self) -> torch.Tensor:
        return self.mat.clone()

    def accuracy(self) -> float:
        correct = torch.diag(self.mat).sum().item()
        total = self.mat.sum().item()
        return 100.0 * correct / max(1, total)

    def per_class_metrics(self):
        mat = self.mat.float()
        tp = torch.diag(mat)
        support = mat.sum(dim=1)       # actual per class (rows)
        pred_sum = mat.sum(dim=0)      # predicted per class (cols)

        precision = tp / pred_sum.clamp_min(1)
        recall    = tp / support.clamp_min(1)
        denom     = (precision + recall).clamp_min(1e-12)
        f1        = 2 * precision * recall / denom

        """
            precision: Tensor [C] in [0,1]
            recall:    Tensor [C] in [0,1]
            f1:        Tensor [C] in [0,1]
            support:   LongTensor [C] (#true samples per class)
        """
        return precision.nan_to_num(0.0), recall.nan_to_num(0.0), f1.nan_to_num(0.0), support.long()

    def macro_f1(self) -> float:
        # Unweighted mean F1 over classes, in percent.
        _, _, f1, _ = self.per_class_metrics()
        return (f1.mean().item() * 100.0)
