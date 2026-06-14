from dataclasses import dataclass

@dataclass
class Config:
    SHOW_PLOTS: bool = True
    plot_flag: int = 1
    n_neighbors: int = 7
    alpha_HNKS: float = 0.8
    lambda_param: float = 0.8
    n_clusters: int = 8
    N_ANCHORS_PER_CLASS: int = 1
    max_iterations: int = 6
    gamma: int = 5