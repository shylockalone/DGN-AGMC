import numpy as np
from sklearn.preprocessing import StandardScaler


def get_domain_datasets(TRANSFER_TASK):
    data_mapping = {
        'A': {
            'data_path': "../data/CWRU/stack_data.npy",
            'labels': "../data/CWRU/labels.npy"
        },
        'B': {
            'data_path': "../data/IMS/stack_data.npy",
            'labels': "../data/IMS/labels.npy"
        },
        'C': {
            'data_path': "../data/DATACase/stack_data.npy",
            'labels': "../data/DATACase/labels.npy"
        }
    }
    
    source_char, target_char = TRANSFER_TASK.split('_to_')
    
    S_info = data_mapping[source_char]
    S_domain_np = np.load(S_info['data_path'])
    S_labels = np.load(S_info['labels'])
    
    T_info = data_mapping[target_char]
    T_domain_np = np.load(T_info['data_path'])
    T_labels = np.load(T_info['labels'])
    
    S_domain_np_reshaped = S_domain_np.reshape(S_domain_np.shape[0], -1)
    T_domain_np_reshaped = T_domain_np.reshape(T_domain_np.shape[0], -1)
    
    scaler = StandardScaler()
    S_domain_np_scaled = scaler.fit_transform(S_domain_np_reshaped)
    T_domain_np_scaled = scaler.transform(T_domain_np_reshaped)
    
    S_domain_np = S_domain_np_scaled.reshape(S_domain_np.shape)
    T_domain_np = T_domain_np_scaled.reshape(T_domain_np.shape)
    
    return S_domain_np, S_labels, T_domain_np, T_labels