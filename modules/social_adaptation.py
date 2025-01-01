"""
Module: social_adaptation.py

Description:
This module implements the social adaptation mechanism, enabling agents to update their 
behavior based on interactions and rewards from other agents. It uses reinforcement 
learning techniques to optimize agent policies over time.

Dependencies:
- gym: For creating reinforcement learning environments.
- torch: For implementing deep reinforcement learning algorithms.
- agent: For accessing agent states and performing policy updates.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random
from modules.agent import Agent

class SocialEnvironment(gym.Env):
    """
    Custom environment for social interactions between agents.
    Implements the OpenAI Gym interface.
    """
    def __init__(self, n_agents: int, state_dim: int, action_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    def calculate_reward(self, agent: Agent, action: np.ndarray, 
                        other_agents: List[Agent]) -> float:
        """
        Calculate social reward based on agent interactions.
        
        Args:
            agent: Current agent
            action: Action taken by the agent
            other_agents: List of other agents in the environment
            
        Returns:
            float: Calculated reward
        """
        # Base reward for action alignment with agent's role
        role_alignment = agent.calculate_role_alignment(action)
        
        # Social influence reward
        influence_reward = sum(other_agent.calculate_influence(agent, action) 
                             for other_agent in other_agents)
        
        # Combine rewards with weights
        total_reward = 0.7 * role_alignment + 0.3 * influence_reward
        return float(total_reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
        action: Action to take, representing agent's behavior vector
            
        Returns:
            Tuple containing:
            - next_state: Updated state after action
            - reward: Social reward obtained
            - done: Whether the episode is complete
            - info: Additional information dictionary
        """
        # Update agent state based on action
        current_state = self.state
        next_state = np.zeros_like(current_state)
        
        # Extract components from state vector based on paper's setup
        influence_level = current_state[0]  # Agent's current influence
        role_vector = current_state[1:4]    # Current role embedding
        social_context = current_state[4:]   # Network context
        
        # Update influence based on action alignment with role
        role_alignment = np.dot(action, role_vector) / (np.linalg.norm(action) * np.linalg.norm(role_vector))
        influence_delta = 0.1 * role_alignment  # Scale factor from paper
        
        # Calculate network effect on influence (from paper's network dynamics)
        network_multiplier = 1.0
        if len(self.other_agents) > 0:
            # Average influence of neighboring agents
            neighbor_influences = np.mean([agent.get_influence() for agent in self.other_agents])
            network_multiplier = 1.0 + 0.2 * (neighbor_influences - influence_level)
        
        # Update influence with both individual and network effects
        new_influence = influence_level + (influence_delta * network_multiplier)
        new_influence = np.clip(new_influence, 0.0, 1.0)  # Bound influence to [0,1]
        
        # Update role vector through soft update
        alpha = 0.1  # Role adaptation rate from paper
        new_role = (1 - alpha) * role_vector + alpha * action[:3]
        new_role = new_role / np.linalg.norm(new_role)  # Normalize
        
        # Update social context based on network state
        new_social_context = self._update_social_context(social_context, action)
        
        # Combine updates into next state
        next_state[0] = new_influence
        next_state[1:4] = new_role
        next_state[4:] = new_social_context
        
        # Calculate reward using the paper's components
        reward = self.calculate_reward(self.current_agent, action, self.other_agents)
        
        # Episode terminates after max_steps (from paper's 1000 time steps)
        done = self.steps >= self.max_steps
        
        # Store additional metrics for analysis
        info = {
            'influence_level': new_influence,
            'role_alignment': role_alignment,
            'network_effect': network_multiplier
        }
        
        self.state = next_state
        self.steps += 1
        
        return next_state, reward, done, info
    
    def _update_social_context(self, current_context: np.ndarray, 
                              action: np.ndarray) -> np.ndarray:
        """
        Helper method to update the social context based on network interactions.
        
        Args:
            current_context: Current social context vector
            action: Agent's action vector
            
        Returns:
            Updated social context vector
        """
        # Extract community structure metrics from context
        clustering_coeff = current_context[0]
        community_vector = current_context[1:]
        
        # Update clustering coefficient based on action similarity with neighbors
        if len(self.other_agents) > 0:
            neighbor_actions = np.array([agent.get_last_action() 
                                       for agent in self.other_agents])
            action_similarities = np.mean([np.dot(action, n_action) / 
                                         (np.linalg.norm(action) * np.linalg.norm(n_action))
                                         for n_action in neighbor_actions])
            
            # Update clustering using paper's community formation rate
            clustering_delta = 0.25 * (action_similarities - clustering_coeff)
            new_clustering = np.clip(clustering_coeff + clustering_delta, 0.0, 1.0)
        else:
            new_clustering = clustering_coeff
        
        # Update community vector through exponential moving average
        beta = 0.15  # Community adaptation rate from paper
        new_community = (1 - beta) * community_vector + beta * action
        new_community = new_community / np.linalg.norm(new_community)  # Normalize
        
        # Combine updates
        new_context = np.zeros_like(current_context)
        new_context[0] = new_clustering
        new_context[1:] = new_community
        
        return new_context

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        raise NotImplementedError

class PolicyNetwork(nn.Module):
    """Neural network for learning social interaction policies"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ExperienceBuffer:
    """Buffer for storing and sampling experience tuples"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

class SocialAdaptationModule:
    """Main class for handling social adaptation through reinforcement learning"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 buffer_size: int = 10000, batch_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.experience_buffer = ExperienceBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = 0.001  # Target network update rate

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return np.random.uniform(-1, 1, size=self.policy_net.network[-1].out_features)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy_net(state_tensor).cpu().numpy()[0]
        return action

    def update(self, state: np.ndarray, action: np.ndarray, 
               reward: float, next_state: np.ndarray, done: bool):
        """
        Update the policy based on experience
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Episode termination flag
        """
        # Store experience
        self.experience_buffer.push(state, action, reward, next_state, done)
        
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # Sample and prepare batch
        states, actions, rewards, next_states, dones = self.experience_buffer.sample(
            self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute target Q values
        with torch.no_grad():
            next_actions = self.target_net(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_actions
            
        # Compute current Q values and loss
        current_q = self.policy_net(states)
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def save_model(self, path: str):
        """Save policy network state"""
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str):
        """Load policy network state"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
