"""
Module: utils.py

Description:
This module provides utility functions for data loading, preprocessing, and metric logging. 
It serves as a support module for the rest of the framework.

Dependencies:
- pandas: For loading and manipulating tabular data.
- os: For file path operations.
- matplotlib: For visualizing results.
"""

import os
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, List, Any, Union
import json

class MetricsLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = {}
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=os.path.join(log_dir, 'simulation.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_metric(self, metric_name: str, value: float, step: int):
        """Records a metric at a specific simulation step."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append((step, value))
        logging.info(f"Step {step}: {metric_name} = {value}")

    def save_metrics(self):
        """Saves all metrics in CSV format."""
        for metric_name, values in self.metrics.items():
            df = pd.DataFrame(values, columns=['step', 'value'])
            df.to_csv(os.path.join(self.log_dir, f'{metric_name}.csv'), index=False)

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Loads a specific dataset from the data directory."""
        file_path = os.path.join(self.data_dir, f'{dataset_name}.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocesses text for analysis."""
        # Implement basic preprocessing
        text = text.lower().strip()
        return text

    @staticmethod
    def preprocess_image(image_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocesses image tensors."""
        # Basic normalization
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        return image_tensor

class NetworkAnalytics:
    @staticmethod
    def calculate_network_metrics(adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Calculates basic network metrics."""
        n_nodes = len(adjacency_matrix)
        density = np.sum(adjacency_matrix) / (n_nodes * (n_nodes - 1))
        
        # Calculate average degree
        degree = np.sum(adjacency_matrix, axis=1)
        avg_degree = np.mean(degree)
        
        return {
            'density': density,
            'avg_degree': avg_degree,
            'n_nodes': n_nodes
        }

    @staticmethod
    def detect_communities(adjacency_matrix: np.ndarray) -> List[int]:
        """
        Implements a simplified version of the Louvain algorithm for community detection.
        
        Args:
            adjacency_matrix: np.ndarray of shape (n_nodes, n_nodes) representing the network
            
        Returns:
            List[int]: Community assignments for each node
        """
        n_nodes = len(adjacency_matrix)
        
        # Initialize each node in its own community
        communities = list(range(n_nodes))
        
        # Calculate initial modularity
        def calculate_modularity(adj_matrix, communities):
            m = np.sum(adj_matrix) / 2  # Total edge weight
            Q = 0
            for i in range(len(adj_matrix)):
                for j in range(len(adj_matrix)):
                    if communities[i] == communities[j]:
                        k_i = np.sum(adj_matrix[i])
                        k_j = np.sum(adj_matrix[j])
                        Q += (adj_matrix[i,j] - (k_i * k_j) / (2 * m))
            return Q / (2 * m)
        
        # First phase: Node assignment optimization
        improvement = True
        while improvement:
            improvement = False
            for node in range(n_nodes):
                # Get neighboring communities
                neighbors = np.where(adjacency_matrix[node] > 0)[0]
                neighbor_communities = {communities[neigh] for neigh in neighbors}
                
                best_modularity = calculate_modularity(adjacency_matrix, communities)
                best_community = communities[node]
                
                # Try moving node to each neighboring community
                for community in neighbor_communities:
                    if community != communities[node]:
                        old_community = communities[node]
                        communities[node] = community
                        new_modularity = calculate_modularity(adjacency_matrix, communities)
                        
                        if new_modularity > best_modularity:
                            best_modularity = new_modularity
                            best_community = community
                        else:
                            communities[node] = old_community
                
                if communities[node] != best_community:
                    communities[node] = best_community
                    improvement = True
        
        # Renumber communities consecutively
        unique_communities = sorted(set(communities))
        community_map = {old: new for new, old in enumerate(unique_communities)}
        communities = [community_map[c] for c in communities]
        
        return communities

def setup_device() -> torch.device:
    """Sets up the computation device (CPU/GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_simulation_state(state: Dict[str, Any], path: str):
    """Saves the simulation state."""
    torch.save(state, path)

def load_simulation_state(path: str) -> Dict[str, Any]:
    """Loads a simulation state."""
    return torch.load(path)
