from dataclasses import dataclass


@dataclass
class Config:
    # --- Experiment Control ---
    cluster_update_limit: int = 4
    num_runs: int = 5
    TRANSFER_TASK: str = 'A_to_B'
    
    # --- Training Hyperparameters ---
    learning_rate: float = 1e-3
    batch_size: int = 128
    pseudo_label_update_freq: int = 16
    epochs: int = 100
    train_ratio: float = 0.25
    SAMPLES_PER_CLASS: int = 16
    use_scheduler: bool = True
    
    # --- Core Model Parameters ---
    in_channels: int = 1
    num_classes: int = 4
    feature_dim: int = 32
    mode_gated: bool = True
    
    # --- Core Loss Modules Weights ---
    w1_clf: float = 1.0
    w2_pseudo: float = 1.0
    confidence_threshold: float = 0.7
    w3_tri: float = 0.001
    triplet_margin: float = 0.5
    triplet_scale: float = 1.0
    k_anchors_per_class: int = 1
    w4_gda: float = 1.0


@dataclass
class HNKS_AS_pre:
    # --- AGMC Hyperparameters ---
    plot_flag: int = 0
    n_neighbors: int = 7
    alpha_HNKS: float = 0.8
    lambda_param: float = 0.8
    k_anchor: int = 4
    n_clusters: int = 4
    max_iterations: int = 6
    gamma: float = 1.0