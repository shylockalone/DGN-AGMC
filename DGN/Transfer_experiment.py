import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader
from Network import ModularGatedTransferNetwork, train_model
from initial import Config
from BalancedBatchSampler import BalancedBatchSampler
from get_datasets import get_domain_datasets

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pro_accuracies = []
    
    for run in range(config.num_runs):
        S_domain_np, S_labels, T_domain_np, T_labels = get_domain_datasets(config.TRANSFER_TASK)
        
        x_s_np = S_domain_np
        y_s_np = S_labels
        
        np.random.seed(32 + run)
        indices = np.random.permutation(len(T_domain_np))
        train_size = int(config.train_ratio * len(T_domain_np))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        x_t_train_np = T_domain_np[train_indices]
        y_t_train_np = T_labels[train_indices]
        x_t_test_np = T_domain_np[test_indices]
        y_t_test_np = T_labels[test_indices]
        
        x_s_tensor = torch.from_numpy(x_s_np).float().unsqueeze(1)
        y_s_tensor = torch.from_numpy(y_s_np).long()
        x_t_train_tensor = torch.from_numpy(x_t_train_np).float().unsqueeze(1)
        y_t_train_tensor = torch.from_numpy(y_t_train_np).long()
        x_t_test_tensor = torch.from_numpy(x_t_test_np).float().unsqueeze(1)
        y_t_test_tensor = torch.from_numpy(y_t_test_np).long()
        
        initial_anchor_indices = []
        for c in range(config.num_classes):
            class_c_indices = np.where(y_t_train_np == c)[0]
            if len(class_c_indices) < config.k_anchors_per_class:
                raise ValueError(f"Class {c} has insufficient samples.")
            chosen_indices = np.random.choice(class_c_indices, size=config.k_anchors_per_class, replace=False)
            initial_anchor_indices.extend(chosen_indices)
        
        initial_anchor_samples_np = x_t_train_np[initial_anchor_indices]
        initial_anchor_samples_tensor = torch.from_numpy(initial_anchor_samples_np).float().unsqueeze(1)
        
        model = ModularGatedTransferNetwork(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        source_dataset = TensorDataset(x_s_tensor, y_s_tensor, torch.arange(len(y_s_tensor)))
        source_sampler = BalancedBatchSampler(y_s_np, config.num_classes, config.SAMPLES_PER_CLASS)
        source_loader_train = DataLoader(source_dataset, batch_sampler=source_sampler)
        
        target_train_dataset = TensorDataset(x_t_train_tensor, y_t_train_tensor, torch.arange(len(y_t_train_tensor)))
        target_sampler = BalancedBatchSampler(y_t_train_np, config.num_classes, config.SAMPLES_PER_CLASS)
        target_loader_train = DataLoader(target_train_dataset, batch_sampler=target_sampler)
        
        target_loader_predict = DataLoader(target_train_dataset, batch_size=config.batch_size, shuffle=False)
        test_dataset = TensorDataset(x_t_test_tensor, y_t_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        model = train_model(
            model=model,
            source_loader=source_loader_train,
            target_loader=target_loader_train,
            target_predict_loader=target_loader_predict,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            initial_anchors=initial_anchor_samples_tensor
        )
        
        model.eval()
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                features = model.feature_extractor(x_test)
                outputs = model.classifier(features)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(y_test.numpy())
        
        accuracy = accuracy_score(all_true_labels, all_predictions)
        pro_accuracies.append(accuracy)