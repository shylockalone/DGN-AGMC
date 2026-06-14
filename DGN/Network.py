import torch
import torch.nn as nn
import torch.nn.functional as F
from initial import Config
from cls_los import lmmd_loss, logexp_triplet_loss, pluggable_clustering
from typing import Tuple, Dict, Any


class DSBN1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_s = nn.BatchNorm1d(num_features)
        self.bn_t = nn.BatchNorm1d(num_features)
        self.target_domain = False
    
    def forward(self, x):
        return self.bn_t(x) if self.target_domain else self.bn_s(x)
    
    def set_target(self, is_target):
        self.target_domain = is_target


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 32):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, stride=1, padding=1),
            DSBN1d(128), nn.LeakyReLU(), nn.Dropout1d(p=0.1), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            DSBN1d(64), nn.LeakyReLU(), nn.Dropout1d(p=0.1), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            DSBN1d(32), nn.LeakyReLU(), nn.Dropout1d(p=0.1), nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_block = nn.Sequential(nn.Linear(32, 32), nn.LeakyReLU(), nn.Linear(32, feature_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptive_pool(self.conv_block(x)).view(x.size(0), -1)
        return self.fc_block(x)


class Classifier(nn.Module):
    def __init__(self, feature_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 32), nn.LeakyReLU(), nn.Dropout(0.3),
                                     nn.Linear(32, num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """Feature-Aware Two-Stream Gating Network (FAGN)"""
    
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.domain_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, feature_dim)
        )
        self.content_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, feature_dim)
        )
        
        self.domain_net[-1].bias.data.fill_(0.0)
        self.content_net[-1].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.domain_net[-1].weight)
        nn.init.xavier_uniform_(self.content_net[-1].weight)
    
    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        fs_mean, ft_mean = torch.mean(f_s, dim=0, keepdim=True), torch.mean(f_t, dim=0, keepdim=True)
        fs_sq_mean, ft_sq_mean = torch.mean(f_s ** 2, dim=0, keepdim=True), torch.mean(f_t ** 2, dim=0, keepdim=True)
        
        discrepancy_info = torch.cat([torch.abs(fs_mean - ft_mean), torch.abs(fs_sq_mean - ft_sq_mean)], dim=1)
        w_transfer = self.domain_net(discrepancy_info)
        
        combined_mean = torch.mean(torch.cat([f_s, f_t], dim=0), dim=0, keepdim=True)
        w_saliency = self.content_net(combined_mean)
        
        raw_weights = w_transfer * w_saliency
        final_gate_weights = F.softmax(raw_weights, dim=1)
        return final_gate_weights.squeeze()


class ModularGatedTransferNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config.in_channels, config.feature_dim)
        self.classifier = Classifier(config.feature_dim, config.num_classes)
        self.gating_network = GatingNetwork(config.feature_dim) if config.mode_gated else None
    
    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> Dict[str, Any]:
        for m in self.feature_extractor.modules():
            if isinstance(m, DSBN1d): m.set_target(False)
        f_s = self.feature_extractor(x_s)
        f_t = self.feature_extractor(x_t)
        
        gate_weights = self.gating_network(f_s.detach(), f_t.detach()) if self.gating_network else None
        f_s_gated = f_s * gate_weights if gate_weights is not None else f_s
        f_t_gated = f_t * gate_weights if gate_weights is not None else f_t
        
        for m in self.feature_extractor.modules():
            if isinstance(m, DSBN1d): m.set_target(True)
        
        return {"f_s": f_s_gated, "p_s": self.classifier(f_s_gated), "f_t": f_t_gated,
                "p_t": self.classifier(f_t_gated), "gate_weights": gate_weights}


def compute_total_loss(outputs: Dict[str, Any], labels: Dict[str, Any], y_t_pseudo: torch.Tensor, config: Config) -> \
Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = torch.tensor(0.0, device=outputs["f_s"].device)
    loss_dict = {}
    
    if config.w1_clf > 0:
        loss_clf = F.cross_entropy(outputs["p_s"], labels["y_s"])
        total_loss += config.w1_clf * loss_clf
        loss_dict["L_clf"] = loss_clf.item()
    
    if config.w2_pseudo > 0:
        p_t_logits = outputs["p_t"]
        p_t_probs = F.softmax(p_t_logits, dim=1)
        max_probs, y_t_pred = torch.max(p_t_probs, dim=1)
        
        high_confidence_mask = (y_t_pred == y_t_pseudo) & (max_probs > config.confidence_threshold)
        
        if high_confidence_mask.sum() > 0:
            loss_psl = F.cross_entropy(p_t_logits[high_confidence_mask], y_t_pseudo[high_confidence_mask])
        else:
            loss_psl = torch.tensor(0.0, device=p_t_logits.device)
        
        total_loss += config.w2_pseudo * loss_psl
        loss_dict["L_pseudo"] = loss_psl.item()
    
    if config.w3_tri > 0 and labels.get("f_t_anchors") is not None:
        loss_tri = logexp_triplet_loss(outputs["f_t"], y_t_pseudo, labels["f_t_anchors"], labels["y_t_anchors"],
                                       config.triplet_margin, config.triplet_scale)
        total_loss += config.w3_tri * loss_tri
        loss_dict["L_tri"] = loss_tri.item()
    
    if config.w4_gda > 0:
        loss_gda = lmmd_loss(outputs["f_s"], labels["y_s"], outputs["f_t"], outputs["p_t"], config.num_classes)
        total_loss += config.w4_gda * loss_gda
        loss_dict["L_gda"] = loss_gda.item()
    
    return total_loss, loss_dict


def update_pseudo_labels_and_anchors(model, data_loader, anchors, config, device):
    model.eval()
    all_f_t, all_y_t = [], []
    with torch.no_grad():
        for x_t, y_t, _ in data_loader:
            all_f_t.append(model.feature_extractor(x_t.to(device)))
            all_y_t.append(y_t.to(device))
    
    pseudo_labels, new_anchors = pluggable_clustering(torch.cat(all_f_t), anchors, torch.cat(all_y_t))
    return pseudo_labels, new_anchors


def compute_core_anchors(anchor_features, num_classes, k_anchors):
    if k_anchors == 1: return anchor_features
    return torch.mean(anchor_features.view(num_classes, k_anchors, -1), dim=1)


def train_model(model, source_loader, target_loader, target_predict_loader, optimizer, scheduler, config, device,
                initial_anchors):
    initial_anchors = initial_anchors.to(device)
    y_t_anchors = torch.arange(config.num_classes, device=device).long()
    f_t_anchors_for_loss = None
    update_counter = 0
    
    for epoch in range(config.epochs):
        if epoch % config.pseudo_label_update_freq == 0 and update_counter < config.cluster_update_limit:
            model.eval()
            with torch.no_grad():
                f_t_anchors_raw = model.feature_extractor(initial_anchors)
                core_anchors = compute_core_anchors(f_t_anchors_raw, config.num_classes, config.k_anchors_per_class)
            pseudo_labels_full, f_t_anchors_for_loss = update_pseudo_labels_and_anchors(model, target_predict_loader,
                                                                                        core_anchors, config, device)
            update_counter += 1
        
        model.train()
        running_loss = 0.0
        iterator = iter(zip(source_loader, target_loader))
        
        for (x_s, y_s, _), (x_t, _, t_indices) in iterator:
            x_s, y_s, x_t, t_indices = x_s.to(device), y_s.to(device), x_t.to(device), t_indices.to(device)
            y_t_pseudo_batch = pseudo_labels_full[t_indices]
            
            optimizer.zero_grad()
            outputs = model(x_s, x_t)
            
            labels_dict = {"y_s": y_s, "f_t_anchors": f_t_anchors_for_loss, "y_t_anchors": y_t_anchors}
            total_loss, _ = compute_total_loss(outputs, labels_dict, y_t_pseudo_batch, config)
            
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
        
        if config.use_scheduler:
            scheduler.step()
    
    return model