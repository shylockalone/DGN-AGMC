import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D

# ==============================================================================
#  Global Style Constants
# ==============================================================================
PUBLICATION_COLORS = [
    '#17becf', '#f4a99b', '#6aaf2d', '#eee484', '#bababa', '#f79421', '#5aafdc', '#802267', '#69d569', '#d493a5'
]
PUBLICATION_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']


# ==============================================================================
#  Algorithm Core Functions
# ==============================================================================
def cal_pairwise_dist(X):
    dist_test = squareform(pdist(X, metric='euclidean'))
    dist = np.square(dist_test)
    return dist


def rbf(dist, t=1.0):
    return np.exp(-(dist / t))


def cal_neighborhood_similarity(dist, nearest_neighbors, n_neighbors, alpha):
    N = dist.shape[0]
    S = np.zeros([N, N])
    neighbor_sets = [set(neighbors) for neighbors in nearest_neighbors]
    rank_lookups = [{neighbor: rank + 1 for rank, neighbor in enumerate(nearest_neighbors[i])} for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            common_neighbors = neighbor_sets[i] & neighbor_sets[j]
            g = len(common_neighbors) / float(n_neighbors - 1) if n_neighbors > 1 else 0.0
            Nrs = 0.0
            if n_neighbors >= 2 and len(common_neighbors) > 0:
                numerator_sum_nrs = 0.0
                for cn_idx in common_neighbors:
                    rank_i = rank_lookups[i].get(cn_idx, 0)
                    rank_j = rank_lookups[j].get(cn_idx, 0)
                    if rank_i > 0 and rank_j > 0:
                        delta_nrs = abs(rank_i - rank_j)
                        term_nrs = (2 * n_neighbors) - rank_i - rank_j - delta_nrs
                        numerator_sum_nrs += term_nrs
                denominator_nrs = float(n_neighbors * (n_neighbors + 1) - 2)
                if denominator_nrs > 0:
                    Nrs = numerator_sum_nrs / denominator_nrs
            Nrs = np.clip(Nrs, 0.0, 1.0)
            similarity = alpha * g + (1 - alpha) * Nrs
            S[i, j] = S[j, i] = similarity
    return S


def cal_rbf_dist_MROS(data, n_neighbors=10, lambda_param=0.5, alpha=0.5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    nearest_neighbors = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    N = data.shape[0]
    max_dist = np.max(neighbor_distances) / 100
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    rbf_dist = rbf(dist, max_dist)
    S = cal_neighborhood_similarity(dist, nearest_neighbors, n_neighbors, alpha)
    W = np.zeros([N, N])
    mutual_pairs = set()
    for i in range(N):
        for j in nearest_neighbors[i]:
            pair = tuple(sorted([i, j]))
            mutual_pairs.add(pair)
    for i, j in mutual_pairs:
        W[i, j] = (1 - lambda_param) * rbf_dist[i, j] + lambda_param * S[i, j]
        W[j, i] = W[i, j]
    return W


def compute_class_centers(data, labels):
    centers = []
    unique_labels = np.sort(np.unique(labels))
    for label in unique_labels:
        if label != -1:
            centers.append(np.mean(data[labels == label], axis=0))
    if len(centers) == 0:
        return np.array([]), unique_labels
    return np.array(centers), unique_labels


def solve_neighbor_assignment_simple(distances, nearest_indices, k):
    m = len(distances)
    z = np.zeros(m)
    if len(nearest_indices) == 0:
        return z
    k_distances = distances[nearest_indices]
    weights = 1.0 / (k_distances + 1e-8)
    weights = weights / np.sum(weights)
    z[nearest_indices] = weights
    return z


def compute_B_affinity(data, anchors, k=4):
    n_samples = data.shape[0]
    n_anchors = anchors.shape[0]
    distances = cdist(data, anchors, metric='euclidean')
    B_affinity = np.zeros((n_samples, n_anchors))
    for i in range(n_samples):
        nearest_indices = np.argsort(distances[i])[:k]
        B_affinity[i, :] = solve_neighbor_assignment_simple(
            distances[i], nearest_indices, k
        )
    return B_affinity


def solve_semi_supervised_factorization(W_mros, B_affinity, n_clusters, gamma):
    n_samples, n_anchors = B_affinity.shape
    
    B_fixed = np.eye(n_clusters)
    dominant_anchor_indices = np.argmax(B_affinity, axis=1)
    T_assignment = B_fixed[dominant_anchor_indices]
    
    D_mros = np.diag(np.sum(W_mros, axis=1))
    L_mros = D_mros - W_mros
    
    val_manifold = np.trace(T_assignment.T @ L_mros @ T_assignment) + 1e-10
    eigvals, eigvecs = eigh(L_mros)
    F_eig = eigvecs[:, 1:n_clusters + 1]
    F_eig = F_eig / (np.linalg.norm(F_eig, axis=1, keepdims=True) + 1e-10)
    val_anchor = np.linalg.norm(F_eig - T_assignment, 'fro') ** 2
    eta = val_anchor / val_manifold
    
    L_mros = eta * L_mros
    I_n = np.eye(n_samples)
    propagator = inv(I_n + gamma * L_mros)
    F = propagator @ T_assignment
    return F


# ======================================================================================
# Visualization Functions
# ======================================================================================

def plot_publication_scatter(reduced_data, labels, ax=None):
    """Plot high-quality scatter plot conforming to top journal standards."""
    try:
        import seaborn as sns
        sns.set_context("paper", font_scale=1.2)
        sns.set_style("ticks")
    except:
        pass
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            c = '#e0e0e0'
            m = '.'
            alpha = 0.3
            zorder = 0
            edgecolor = 'none'
            s_size = 20
            linewidth = 0
        else:
            c = PUBLICATION_COLORS[int(label) % len(PUBLICATION_COLORS)]
            m = PUBLICATION_MARKERS[int(label) % len(PUBLICATION_MARKERS)]
            alpha = 0.8
            zorder = 2
            edgecolor = 'black'
            linewidth = 0.5
            s_size = 50
        
        mask = (labels == label)
        ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                   c=c, marker=m, s=s_size, alpha=alpha,
                   edgecolors=edgecolor, linewidths=linewidth, zorder=zorder)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='in', width=0.8, length=4)
    ax.grid(False)
    
    return ax


def plot_standalone_legend(unique_labels, show_anchors=False, show_centers=False):
    """Generate legend on a standalone canvas."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    legend_elements = []
    valid_labels = [l for l in unique_labels if l != -1]
    
    for label in valid_labels:
        c = PUBLICATION_COLORS[int(label) % len(PUBLICATION_COLORS)]
        m = PUBLICATION_MARKERS[int(label) % len(PUBLICATION_MARKERS)]
        legend_elements.append(
            Line2D([0], [0], marker=m, color='w', label=f'Cluster {int(label)}',
                   markerfacecolor=c, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        )
    
    if show_anchors:
        legend_elements.append(
            Line2D([0], [0], marker='*', color='w', label='Anchor',
                   markerfacecolor='gold', markersize=14, markeredgecolor='black', markeredgewidth=1)
        )
    if show_centers:
        legend_elements.append(
            Line2D([0], [0], marker='X', color='w', label='Center',
                   markerfacecolor='#d62728', markersize=12, markeredgecolor='white', markeredgewidth=1)
        )
    
    ax.legend(handles=legend_elements, loc='center', ncol=5, frameon=False,
              fontsize=11, columnspacing=1.5, handletextpad=0.2)
    plt.show()


# ======================================================================================
# Main Iteration Logic
# ======================================================================================

def iterative_clustering(W_mros, plot_flag, reduced_data, true_labels, initial_anchor_data, k_anchor,
                         max_iterations=5, n_clusters=4, gamma=0.5):
    anchor_data = np.copy(initial_anchor_data)
    centers_history = []
    labels_history = []
    
    print(f"Starting iterative clustering (Max iter: {max_iterations})...")
    Ground_truth_flag = False
    
    for iteration in range(max_iterations):
        previous_anchors = np.copy(anchor_data)
        
        B_affinity = compute_B_affinity(reduced_data, anchor_data, k_anchor)
        F = solve_semi_supervised_factorization(W_mros, B_affinity, n_clusters=n_clusters, gamma=gamma)
        F_normalized = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-10)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        raw_labels = kmeans.fit_predict(F_normalized)
        
        new_centers, unique_raw_labels = compute_class_centers(reduced_data, raw_labels)
        
        if len(new_centers) != n_clusters:
            print(f"  Warning: K-means found {len(new_centers)} clusters. Stopping.")
            return labels_history[-1] if labels_history else raw_labels, anchor_data
        
        dist_matrix = cdist(previous_anchors, new_centers)
        prev_anchor_indices, new_center_indices = linear_sum_assignment(dist_matrix)
        
        map_raw_to_stable = {unique_raw_labels[c_idx]: p_idx for p_idx, c_idx in
                             zip(prev_anchor_indices, new_center_indices)}
        current_labels = np.array([map_raw_to_stable.get(l, -1) for l in raw_labels])
        
        labels_history.append(current_labels.copy())
        
        updated_anchor_data = np.zeros_like(previous_anchors)
        for stable_idx, new_center_idx in zip(prev_anchor_indices, new_center_indices):
            updated_anchor_data[stable_idx] = new_centers[new_center_idx]
        
        centers_history.append(updated_anchor_data.copy())
        
        is_converged = False
        if iteration > 0:
            label_diff = np.sum(current_labels != labels_history[-2])
            print(f"  > Iteration {iteration + 1}: {label_diff} labels changed compared to previous iteration.")
            
            if label_diff == 0:
                print(f"  > Converged at iteration {iteration + 1}.")
                is_converged = True
                anchor_data = updated_anchor_data
        else:
            print(f"  > Iteration 1: Initial assignment completed.")
        
        anchor_data = updated_anchor_data
        
        if plot_flag == 1:
            print(f"--- Plotting Iteration {iteration + 1} ---")
            
            fig1, ax1 = plt.subplots(figsize=(8, 3.5), dpi=120)
            plot_publication_scatter(reduced_data, current_labels, ax=ax1)
            
            ax1.scatter(previous_anchors[:, 0], previous_anchors[:, 1],
                        c='gold', marker='*', s=300, edgecolors='black', linewidths=1.2, zorder=10)
            ax1.scatter(new_centers[:, 0], new_centers[:, 1],
                        c='#d62728', marker='X', s=150, edgecolors='white', linewidths=0.5, zorder=11)
            
            if not Ground_truth_flag:
                fig2, ax2 = plt.subplots(figsize=(8, 3.5), dpi=120)
                plot_publication_scatter(reduced_data, true_labels, ax=ax2)
                Ground_truth_flag = True
            
            plt.show()
        
        if is_converged:
            break
    
    return current_labels, anchor_data