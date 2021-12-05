from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()

        self.temp = temp
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos_sim(x, y) / self.temp


class ContrastiveLoss(nn.Module):
    """
    Loss Function adapted from https://arxiv.org/abs/2104.08821
    """

    def __init__(self, temp: int = 0.05, hard_negative_weight: float = 0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.sim = Similarity(temp)
        self.hard_negative_weight = hard_negative_weight
        # self.loss_fn = nn.CrossEntropyLoss

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        hard_negative: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        assert input.shape == target.shape

        cos_sim = self.sim(input.unsqueeze(1), target.unsqueeze(0))

        if hard_negative is not None:
            input_negative_sim = self.sim(
                input.unsqueeze(1), hard_negative.unsqueeze(0)
            )
            cos_sim = torch.cat([cos_sim, input_negative_sim], dim=1)

            weights = torch.tensor(
                [
                    [0.0] * (cos_sim.size(-1) - input_negative_sim.size(-1))
                    + [0.0] * i
                    + [self.hard_negative_weight]
                    + [0.0] * (input_negative_sim.size(-1) - i - 1)
                    for i in range(input_negative_sim.size(-1))
                ],
                device=input.device,
            )
            cos_sim = cos_sim + weights

        labels = torch.arange(cos_sim.size(0), device=input.device).long()
        loss = F.cross_entropy(cos_sim, labels)

        return loss


class AlignmentLoss(nn.Module):
    def __init__(self, alpha: int = 2):
        super(AlignmentLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.mean(
            torch.norm(x - y, p=2, dim=1).pow(self.alpha)
        )


class UniformityLoss(nn.Module):
    def __init__(self):
        super(UniformityLoss, self).__init__()

    def forward(self, x: torch.Tensor):
        return torch.log(
            torch.mean(
                torch.exp(-2 * torch.pdist(x, p=2).pow(2))
                )
            )
