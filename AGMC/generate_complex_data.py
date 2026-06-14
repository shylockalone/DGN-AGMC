import numpy as np
import matplotlib.pyplot as plt


def generate_custom_dataset(
        random_state=42,
        blob_configs=[
            {'n_samples': 190, 'center': (-2, 1.5), 'std_devs': (1.2, 0.3), 'angle': 15},
            {'n_samples': 190, 'center': (-3.2, -1.7), 'std_devs': (0.3, 1.0), 'angle': -30},
            {'n_samples': 190, 'center': (0, -1.9), 'std_devs': (0.3, 1.0), 'angle': -105},
            {'n_samples': 170, 'center': (0, 0), 'std_devs': (0.4, 0.4), 'angle': 0},
        ],
        shape_configs=[
            {'type': 'arc', 'n_samples': 190, 'noise': 0.1, 'center': (3.7, 1.5), 'radius': 1.5,
             'angle_range': (np.pi / 2, 3 * np.pi / 2)},
            {'type': 'arc', 'n_samples': 190, 'noise': 0.1, 'center': (4, -0.5), 'radius': 1.5,
             'angle_range': (-np.pi / 2, np.pi / 2)},
            {'type': 'circle', 'n_samples': 220, 'noise': 0.12, 'center': (8.5, 0), 'radius': 1.5},
            {'type': 'curve', 'n_samples': 200, 'noise': 0.15, 'x_range': (3, 9),
             'coeffs': {'a': -0.05, 'b': 0.1, 'c': 0.5, 'k': -4.2}}
        ]
):
    np.random.seed(random_state)
    X_blobs_list, y_blobs_list = [], []
    
    for i, config in enumerate(blob_configs):
        n_samples, center, std_devs, angle_deg = config['n_samples'], config['center'], config['std_devs'], config[
            'angle']
        angle_rad = np.deg2rad(angle_deg)
        cov_unrotated = np.diag([std ** 2 for std in std_devs])
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        cov_rotated = rotation_matrix @ cov_unrotated @ rotation_matrix.T
        X_blob = np.random.multivariate_normal(mean=center, cov=cov_rotated, size=n_samples)
        X_blobs_list.append(X_blob)
        y_blobs_list.append(np.full(n_samples, i))
    
    X_blobs = np.vstack(X_blobs_list)
    y_blobs = np.concatenate(y_blobs_list)
    
    X_shapes_list, y_shapes_list = [], []
    base_label = len(blob_configs)
    
    for i, config in enumerate(shape_configs):
        shape_type, n_samples, noise = config['type'], config['n_samples'], config['noise']
        label = base_label + i
        X_shape = None
        
        if shape_type in ['arc', 'circle']:
            center, radius = config['center'], config['radius']
            angle_range = config.get('angle_range', (0, 2 * np.pi))
            theta = np.linspace(angle_range[0], angle_range[1], n_samples)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            X_shape = np.vstack([x, y]).T
        elif shape_type == 'curve':
            x_range, coeffs = config['x_range'], config['coeffs']
            x = np.linspace(x_range[0], x_range[1], n_samples)
            h = (x_range[0] + x_range[1]) / 2
            x_centered = x - h
            y = (coeffs['a'] * x_centered ** 3 + coeffs['b'] * x_centered ** 2 + coeffs['c'] * x_centered + coeffs['k'])
            X_shape = np.vstack([x, y]).T
        
        if X_shape is not None:
            X_shape += np.random.randn(n_samples, 2) * noise
            X_shapes_list.append(X_shape)
            y_shapes_list.append(np.full(n_samples, label))
    
    X = np.vstack([X_blobs] + X_shapes_list)
    y = np.concatenate([y_blobs] + y_shapes_list)
    
    print(f"Dataset generated. Total samples: {X.shape[0]}, Total clusters: {len(np.unique(y))}")
    return X, y


def plot_dataset(X, y):
    plt.figure(figsize=(14, 9))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    unique_labels = np.unique(y)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = (y == label)
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], label=f'Cluster {label} (n={np.sum(mask)})', s=15, alpha=0.8)
    
    plt.title('Generated Final Complex Dataset', fontsize=16)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X_custom, y_custom = generate_custom_dataset(random_state=42)
    plot_dataset(X_custom, y_custom)