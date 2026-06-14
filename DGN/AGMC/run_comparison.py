import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Scikit-learn ---
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.decomposition import PCA

# --- Custom Modules ---
from generate_complex_data import generate_custom_dataset
from scipy.optimize import linear_sum_assignment
from initial import Config

try:
    from AGMC import cal_rbf_dist_MROS, iterative_clustering
except ImportError:
    print("Error: Cannot import AGMC.py. Please ensure it is in the same directory.")
    exit()


def plot_publication_scatter(reduced_data, labels, title="Clustering Result"):
    """Plot high-quality scatter plot conforming to top journal standards."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors_hex = ['#17becf', '#f4a99b', '#6aaf2d', '#eee484', '#bababa', '#f79421', '#5aafdc', '#802267', '#69d569',
                  '#d493a5']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label == -1:
            color, marker, class_name = '#bababa', '.', 'Noise'
        else:
            color = colors_hex[int(label % len(colors_hex))]
            marker = markers[int(label % len(markers))]
            class_name = f'Cluster {int(label)}'
        
        mask = (labels == label)
        ax.scatter(
            reduced_data[mask, 0], reduced_data[mask, 1],
            c=color, marker=marker, label=class_name,
            s=25, alpha=0.8, edgecolors='black', linewidths=0.5
        )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=10)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    
    if 1 < len(unique_labels) < 15:
        num_columns = int(np.ceil(len(unique_labels) / 2))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=num_columns, frameon=True, fontsize=8,
                  markerscale=1.2)
    
    plt.tight_layout()
    return fig


def clustering_accuracy(y_true, y_pred):
    """Calculate clustering accuracy (ACC) using the Hungarian algorithm."""
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    n_correct = contingency[row_ind, col_ind].sum()
    return n_correct / len(y_true)


def prepare_semisupervised_data(X, y_true, n_anchors_per_class=1, random_state=42):
    """Prepare initial anchors for semi-supervised algorithms."""
    np.random.seed(random_state)
    unique_labels = np.unique(y_true)
    anchor_indices = []
    
    for label in unique_labels:
        label_indices = np.where(y_true == label)[0]
        chosen_indices = np.random.choice(label_indices, size=n_anchors_per_class, replace=False)
        anchor_indices.extend(chosen_indices)
    
    anchor_indices = np.array(anchor_indices)
    initial_anchors = X[anchor_indices]
    
    print(f"Extracted {len(initial_anchors)} anchors.")
    return initial_anchors


def evaluate_clustering(X, y_true, y_pred, algorithm_name):
    """Calculate and return all clustering evaluation metrics."""
    y_true_valid, y_pred_valid = y_true, y_pred
    
    if -1 in y_pred:
        valid_indices = y_pred != -1
        y_true_valid, y_pred_valid = y_true[valid_indices], y_pred[valid_indices]
    
    if len(np.unique(y_pred_valid)) <= 1:
        return {'ACC': 0, 'ARI': 0, 'NMI': 0, 'Silhouette': -1}
    
    acc = clustering_accuracy(y_true_valid, y_pred_valid)
    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)
    silhouette = silhouette_score(X[y_pred != -1], y_pred_valid)
    
    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'Silhouette': silhouette}


if __name__ == "__main__":
    
    RANDOM_STATE = 18
    
    # 1. Load custom complex dataset directly
    X, y_true = generate_custom_dataset(random_state=42)
    
    # Dataset sampling
    sample_ratio = 0.7
    np.random.seed(RANDOM_STATE)
    indices = np.random.permutation(len(X))
    train_size = int(sample_ratio * len(X))
    train_indices = indices[:train_size]
    
    X = X[train_indices]
    y_true = y_true[train_indices]
    
    n_clusters = Config.n_clusters
    
    # 2. Prepare semi-supervised anchor data
    initial_anchors = prepare_semisupervised_data(
        X, y_true,
        n_anchors_per_class=Config.N_ANCHORS_PER_CLASS,
        random_state=RANDOM_STATE
    )
    
    # 3. Minimal comparison dictionary: K-Means and AGMC only
    models = {
        "K-Means": KMeans(n_clusters=n_clusters, n_init='auto', random_state=RANDOM_STATE),
        "AGMC": "custom"
    }
    
    results_list = []
    
    # 4. Execute evaluation
    for name, model in models.items():
        print(f"\n--- Running algorithm: {name} ---")
        try:
            if name == "AGMC":
                W_mros = cal_rbf_dist_MROS(X, n_neighbors=Config.n_neighbors, lambda_param=Config.lambda_param,
                                           alpha=Config.alpha_HNKS)
                y_pred, _ = iterative_clustering(
                    W_mros=W_mros, plot_flag=Config.plot_flag, reduced_data=X, true_labels=y_true,
                    initial_anchor_data=initial_anchors, k_anchor=Config.n_clusters,
                    max_iterations=Config.max_iterations, n_clusters=Config.n_clusters, gamma=Config.gamma
                )
            else:
                y_pred = model.fit_predict(X)
            
            metrics = evaluate_clustering(X, y_true, y_pred, name)
            results_list.append({"name": name, "labels": y_pred, "metrics": metrics})
            print(f"Completed. ACC: {metrics['ACC']:.4f}, ARI: {metrics['ARI']:.4f}, NMI: {metrics['NMI']:.4f}")
        
        except Exception as e:
            print(f"Error occurred while running {name}: {e}")
    
    # 5. Print results
    results_df = pd.DataFrame([
        {'Algorithm': res['name'], **res['metrics']} for res in results_list
    ]).sort_values(by='ACC', ascending=False)
    
    print("\n\n" + "=" * 70)
    print(" " * 20 + "Experimental Results Summary (Sorted by ACC)")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))
    print("=" * 70)
    
    # 6. Visualization
    if X.shape[1] > 2:
        print("\nData is high-dimensional, applying PCA for visualization...")
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_vis = pca.fit_transform(X)
    else:
        X_vis = X
    
    fig_gt = plot_publication_scatter(X_vis, y_true, title="Ground Truth")
    if not Config.SHOW_PLOTS:
        plt.close(fig_gt)
    
    sorted_results = sorted(results_list, key=lambda x: x['metrics']['ACC'], reverse=True)
    
    for res in sorted_results:
        fig_res = plot_publication_scatter(X_vis, res['labels'], title=res['name'])
        if not Config.SHOW_PLOTS:
            plt.close(fig_res)
    
    if Config.SHOW_PLOTS:
        print("\nDisplaying plots... (Please close the windows manually to exit)")
        plt.show()