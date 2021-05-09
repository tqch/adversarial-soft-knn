import torch
import torch.nn as nn
import torch.nn.functional as F


class ASKLoss(nn.Module):
    """
    Adversarial Soft K-nearest neighbor loss
    """

    def __init__(
            self,
            reduction="mean",
            temperature=1,
            metric="l2"
    ):
        super(ASKLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.metric = metric

    @staticmethod
    def pairwise_l2_distance(x, y, x_other=None):
        """
        x_i,y_j in R^D
        for i=1..M,j=1..N, calculate ||x_i-y_j||^2 => O(3*M*N*D)
        is equivalent to
          for i=1..M ||x_i||^2
        + for i=1..M,j=1..N <x_i,y_j>
        + for j=1..N ||y_j||^2
        _______________________
        O(2*(M+M*N+N)*D+2*M*N)
        """
        dist_matrix = (x.pow(2).sum(dim=1)[:, None] - 2 * x @ y.T + y.pow(2).sum(dim=1)[None, :]).sqrt()
        dist_orig = None
        if x_other is not None:
            dist_orig = (x - x_other).pow(2).sum(dim=1).sqrt()
        return dist_matrix, dist_orig

    @staticmethod
    def pairwise_cosine_similarity(x, y, x_other=None):
        similarity_matrix = x @ y.T / y.pow(2).sum(dim=1)[None, :].sqrt()
        similarity_matrix = 1 / x.pow(2).sum(dim=1)[:, None].sqrt() * similarity_matrix
        similarity_orig = None
        if x_other is not None:
            similarity_orig = (x * x_other).sum(dim=1) / \
                              x.pow(2).sum(dim=1).sqrt() / x_other.pow(2).sum(dim=1).sqrt()
        return similarity_matrix, similarity_orig

    def forward(self, x, y, x_ref, y_ref, x_other=None):
        if self.metric == "l2":
            score_matrix, score_orig = -self.pairwise_l2_distance(x, x_ref, x_other)
        if self.metric == "cosine":
            score_matrix, score_orig = self.pairwise_cosine_similarity(x, x_ref, x_other)
        if score_orig is not None:
            score_matrix = F.softmax(torch.cat([
                score_matrix, score_orig[:, None]
            ], dim=1) / self.temperature, dim=1)
        else:
            score_matrix = F.softmax(score_matrix, dim=1)
        soft_nns = torch.zeros(x.size(0), 10).to(x)
        for i in range(10):
            if (y_ref == i).sum().item() == 0:
                soft_nns[:, i] += 1e-6
            else:
                if score_orig is not None:
                    soft_nns[:, i] += score_matrix[:, :-1][:, y_ref == i].sum(dim=1) + 1e-6
                else:
                    soft_nns[:, i] += score_matrix[:, y_ref == i].sum(dim=1) + 1e-6
        if score_orig is not None:
            soft_nns[range(x.size(0)), y] += score_matrix[:, -1]
        log_soft_nns = torch.log(soft_nns)
        return F.nll_loss(log_soft_nns, y, reduction=self.reduction)
