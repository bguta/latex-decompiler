import torch.nn.functional as F
import torch

def loss_fn(gt, pr):
    """
    pr: probability distribution return by model
            [B, MAX_LEN, voc_size]
    gt: target formulas
            [B, MAX_LEN]
    """
    padding = torch.ones_like(gt) * 0
    mask = (gt != padding)
    targets = gt.masked_select(mask)
    logits = pr.masked_select(
    mask.unsqueeze(2).expand(-1, -1, pr.size(2))
    ).contiguous().view(-1, pr.size(2))
    logits = torch.log(logits)
    assert logits.size(0) == targets.size(0), f"{logits.size()} and {targets.size()} are incompatible"
    assert targets.nelement() != 0, f"targets is empty {targets}"
    loss = F.nll_loss(input=logits, target=targets)
    return loss
