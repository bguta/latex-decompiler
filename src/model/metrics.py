import torch.nn.functional as F
import torch

def loss_fn(gt, pr):
    """
    pr: probability distribution return by model
            [B, MAX_LEN, voc_size]
    gt: target formulas
            [B, MAX_LEN]
    """
    #padding = torch.zeros_like(gt)
    mask = gt.gt(0)
    targets = torch.masked_select(gt, mask) # [len_equation]
    logits = torch.masked_select(pr, mask.unsqueeze(2).expand(-1, -1, pr.size(2))).contiguous().view(-1, pr.size(2)) # [len_equation, voc_size]
    logits = torch.log(logits)
    assert logits.size(0) == targets.size(0), f"{logits.size()} and {targets.size()} are incompatible"
    assert targets.nelement() != 0, f"targets is empty {targets}"
    loss = F.nll_loss(input=logits, target=targets)
    return loss
