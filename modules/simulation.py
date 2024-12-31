"""
Module: simulation.py

Description:
This module configures and runs the social network simulation. It initializes agents, defines 
interaction rules, and executes iterations to simulate network evolution. Results are logged 
for analysis.

Dependencies:
- agent: For initializing and managing agents.
- social_adaptation: For interaction rules and feedback mechanisms.
- utils: For logging and performance tracking.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
import networkx as nx
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

from modules.agent import Agent
from modules.social_adaptation import SocialAdaptationModule
from modules.utils import MetricsLogger

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    num_agents: int
    embedding_dim: int
    action_dim: int
    max_iterations: int
    batch_size: int
    log_interval: int
    checkpoint_interval: int
    memory_capacity: int
    num_concepts: int
    modality_config: Dict[str, Any]

class NetworkMetrics:
    """Calculates and tracks network-level metrics"""
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {
            'density': [],
            'clustering_coefficient': [],
            'average_path_length': [],
            'modularity': [],
            'influence_distribution': []
        }

    def calculate_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate network metrics
        
        Args:
            G: NetworkX graph representing the social network
            
        Returns:
            Dict of calculated metrics
        """
        metrics = {}
        
        # Basic network metrics
        metrics['density'] = nx.density(G)
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        
        # Connected components analysis
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        
        try:
            metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
        except nx.NetworkXError:
            metrics['average_path_length'] = float('inf')
            
        # Community detection
        communities = nx.community.greedy_modularity_communities(G)
        metrics['modularity'] = nx.community.modularity(G, communities)
        
        return metrics

    def update(self, G: nx.Graph) -> None:
        """Update metrics history with current network state"""
        metrics = self.calculate_metrics(G)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

class Simulation:
    """Main simulation class for running social network evolution"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agents
        self.agents: Dict[str, Agent] = {}
        self._initialize_agents()
        
        # Initialize network structure
        self.network = nx.Graph()
        self._initialize_network()
        
        # Setup metrics tracking
        self.metrics = NetworkMetrics()
        self.logger = MetricsLogger("simulation_logs")
        
        self.current_iteration = 0

    def _initialize_agents(self) -> None:
        """Initialize simulation agents"""
        for i in range(self.config.num_agents):
            agent_config = {
                'modality_config': self.config.modality_config,
                'action_dim': self.config.action_dim,
                'hidden_dim': self.config.embedding_dim
            }
            
            agent = Agent(
                agent_id=f"agent_{i}",
                config=agent_config,
                embedding_dim=self.config.embedding_dim,
                memory_capacity=self.config.memory_capacity,
                num_concepts=self.config.num_concepts
            )
            self.agents[agent.agent_id] = agent

    def _initialize_network(self) -> None:
        """Initialize network structure with agents"""
        # Add all agents as nodes
        for agent_id in self.agents.keys():
            self.network.add_node(agent_id)
        
        # Initialize random connections
        for agent_id in self.agents.keys():
            num_connections = np.random.randint(1, 5)
            possible_connections = list(set(self.agents.keys()) - {agent_id})
            if possible_connections:
                connections = np.random.choice(
                    possible_connections,
                    size=min(num_connections, len(possible_connections)),
                    replace=False
                )
                for target_id in connections:
                    self.network.add_edge(agent_id, target_id, weight=0.1)

    def step(self) -> Dict[str, float]:
        """
        Execute one simulation step
        
        Returns:
            Dict of metrics for the current step
        """
        # Process batch of interactions
        interactions = self._sample_interactions(self.config.batch_size)
        
        for agent_id, target_id in interactions:
            agent = self.agents[agent_id]
            target = self.agents[target_id]
            
            # Get current states
            agent_state = agent.state.embedding
            target_state = target.state.embedding
            
            # Agent selects and performs action
            action = agent.select_action(state=agent_state)
            
            # Calculate interaction outcome
            reward = self._calculate_interaction_reward(agent, target, action)
            
            # Update agent states
            next_state = self._get_next_state(agent_state, action, reward)
            agent.update(agent_state, action, reward, next_state, False)
            
            # Update network structure
            self._update_network_connection(agent_id, target_id, reward)

        # Update network metrics
        self.metrics.update(self.network)
        current_metrics = {k: v[-1] for k, v in self.metrics.metrics_history.items()}
        
        # Log metrics
        if self.current_iteration % self.config.log_interval == 0:
            self.logger.log_metric("network_metrics", current_metrics, 
                                 self.current_iteration)
        
        # Save checkpoint if needed
        if self.current_iteration % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
            
        self.current_iteration += 1
        return current_metrics

    def _sample_interactions(self, batch_size: int) -> List[Tuple[str, str]]:
        """Sample random interactions between connected agents"""
        interactions = []
        edges = list(self.network.edges())
        
        if edges:
            sampled_edges = np.random.choice(
                len(edges), 
                size=min(batch_size, len(edges)), 
                replace=False
            )
            interactions = [edges[i] for i in sampled_edges]
            
        return interactions

    def _calculate_interaction_reward(self, agent: Agent, target: Agent, 
                                   action: torch.Tensor) -> float:
        """Calculate reward for an interaction"""
        # Base reward from role alignment
        role_alignment = agent.calculate_role_alignment(action)
        
        # Social influence component
        influence = agent.calculate_influence(target, action)
        
        # Combine components
        reward = 0.7 * role_alignment + 0.3 * influence
        return float(reward)

    def _get_next_state(self, state: torch.Tensor, action: torch.Tensor, 
                       reward: float) -> torch.Tensor:
        """Calculate next state based on current state, action, and reward"""
        # Simple state transition for now
        next_state = state + 0.1 * action
        return next_state

    def _update_network_connection(self, agent_id: str, target_id: str, 
                                 reward: float) -> None:
        """Update network connection strength based on interaction outcome"""
        current_weight = self.network[agent_id][target_id]['weight']
        # Update weight using exponential moving average
        alpha = 0.1
        new_weight = (1 - alpha) * current_weight + alpha * (reward + 1) / 2
        self.network[agent_id][target_id]['weight'] = new_weight

    def run(self, num_iterations: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Run simulation for specified number of iterations
        
        Args:
            num_iterations: Number of iterations to run (default: config.max_iterations)
            
        Returns:
            Dict of metrics history
        """
        iterations = num_iterations or self.config.max_iterations
        
        for _ in range(iterations):
            self.step()
            
        return self.metrics.metrics_history

    def save_checkpoint(self) -> None:
        """Save simulation state checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save network structure
        nx.write_gexf(self.network, 
                     checkpoint_dir / f"network_{self.current_iteration}.gexf")
        
        # Save agents
        for agent_id, agent in self.agents.items():
            agent.save_state(
                str(checkpoint_dir / f"{agent_id}_{self.current_iteration}.pt")
            )
            
        # Save metrics
        with open(checkpoint_dir / f"metrics_{self.current_iteration}.json", 'w') as f:
            json.dump(self.metrics.metrics_history, f)

    def load_checkpoint(self, iteration: int) -> None:
        """Load simulation state from checkpoint"""
        checkpoint_dir = Path("checkpoints")
        
        # Load network structure
        self.network = nx.read_gexf(
            checkpoint_dir / f"network_{iteration}.gexf"
        )
        
        # Load agents
        for agent_id in self.agents.keys():
            self.agents[agent_id].load_state(
                str(checkpoint_dir / f"{agent_id}_{iteration}.pt")
            )
            
        # Load metrics
        with open(checkpoint_dir / f"metrics_{iteration}.json", 'r') as f:
            self.metrics.metrics_history = json.load(f)
            
        self.current_iteration = iteration
