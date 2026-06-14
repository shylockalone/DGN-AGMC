import torch
import torch.nn.functional as F
from initial import HNKS_AS_pre
from typing import Tuple

try:
    from AGMC import cal_rbf_dist_MROS, iterative_clustering
except ImportError:
    pass


def pluggable_clustering(f_t: torch.Tensor, anchor_input: torch.Tensor, true_labels: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    config = HNKS_AS_pre()
    
    f_t_np = f_t.cpu().detach().numpy()
    anchor_input_np = anchor_input.cpu().detach().numpy()
    true_labels_np = true_labels.cpu().detach().numpy()
    
    W_mros = cal_rbf_dist_MROS(
        f_t_np,
        n_neighbors=config.n_neighbors,
        lambda_param=config.lambda_param,
        alpha=config.alpha_HNKS
    )
    
    final_labels, new_anchors_np = iterative_clustering(
        W_mros=W_mros, plot_flag=config.plot_flag, reduced_data=f_t_np, true_labels=true_labels_np,
        initial_anchor_data=anchor_input_np, k_anchor=config.n_clusters,
        max_iterations=config.max_iterations, n_clusters=config.n_clusters, gamma=config.gamma
    )
    
    final_labels_tensor = torch.tensor(final_labels, dtype=torch.long, device=f_t.device)
    new_anchors_tensor = torch.tensor(new_anchors_np, dtype=torch.float32, device=f_t.device)
    
    return final_labels_tensor, new_anchors_tensor


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + 1e-6) + 1e-6
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / band + 1e-6) for band in bandwidth_list]
    return sum(kernel_val)


def lmmd_loss(f_s: torch.Tensor, y_s: torch.Tensor, f_t: torch.Tensor, p_t_logits: torch.Tensor,
              num_classes: int) -> torch.Tensor:

    batch_size_s = f_s.size(0)
    batch_size_t = f_t.size(0)
    
    y_s_onehot = F.one_hot(y_s, num_classes=num_classes).float()  # (batch_size_s, C)
    weight_s = y_s_onehot / (torch.sum(y_s_onehot, dim=0, keepdim=True) + 1e-6)
    
    p_t_probs = F.softmax(p_t_logits, dim=1)  # (batch_size_t, C)
    weight_t = p_t_probs / (torch.sum(p_t_probs, dim=0, keepdim=True) + 1e-6)
    
    kernel_matrix = gaussian_kernel(f_s, f_t)
    
    K_ss = kernel_matrix[:batch_size_s, :batch_size_s]
    K_tt = kernel_matrix[batch_size_s:, batch_size_s:]
    K_st = kernel_matrix[:batch_size_s, batch_size_s:]
    
    loss = torch.tensor(0.0, device=f_s.device)
    
    for c in range(num_classes):
        w_s_c = weight_s[:, c].unsqueeze(1)  # (batch_size_s, 1)
        w_t_c = weight_t[:, c].unsqueeze(1)  # (batch_size_t, 1)
        
        term1 = torch.sum(w_s_c @ w_s_c.t() * K_ss)
        term2 = torch.sum(w_t_c @ w_t_c.t() * K_tt)
        term3 = 2.0 * torch.sum(w_s_c @ w_t_c.t() * K_st)
        
        loss += (term1 + term2 - term3)
    
    return loss / num_classes


def logexp_triplet_loss(f_t: torch.Tensor, y_t_pseudo: torch.Tensor, f_t_anchors: torch.Tensor,
                        y_t_anchors: torch.Tensor, margin: float = 0.5, scale: float = 1.0) -> torch.Tensor:
    """
    Strict implementation of Equation (13) from the paper.
    Computes smooth softplus triplet loss accumulated over all constituent anchors.
    """
    # d(f_i^t, f_j^a): Shape (batch_size, n_anchors)
    dist_matrix = torch.cdist(f_t, f_t_anchors, p=2.0)
    
    # Indicator function mapping: Shape (batch_size, n_anchors)
    # 1 if pseudo_label == anchor_label else 0
    indicator_matrix = (y_t_pseudo.unsqueeze(1) == y_t_anchors.unsqueeze(0)).float()
    
    # (2 * I - 1): Positive anchors become +1, negative anchors become -1
    sign_matrix = 2.0 * indicator_matrix - 1.0
    
    # Scaled exponential inner term: s_cal * ((2*I - 1)*d + delta)
    inner_term = scale * (sign_matrix * dist_matrix + margin)
    
    # Summing exponentials over all anchors (dim=1)
    sum_exp = torch.sum(torch.exp(inner_term), dim=1)
    
    # Log[1 + sum(exp(...))] divided by s_cal
    loss_per_sample = torch.log(1.0 + sum_exp) / scale
    
    return loss_per_sample.mean()