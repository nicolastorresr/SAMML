"""
Script: run_simulation.py

Description:
This script serves as the main entry point for running simulations. It allows users to 
configure parameters, initialize the simulation environment, and save results for analysis.

Dependencies:
- simulation: For configuring and executing the simulation.
- utils: For saving and visualizing simulation results.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any
import logging
import torch
import numpy as np

from modules.simulation import Simulation
from modules.utils import MetricsLogger, NetworkAnalytics, save_simulation_state
from modules.agent import Agent

def parse_args():
    """Parse command line arguments for simulation configuration."""
    parser = argparse.ArgumentParser(description='Run SAMML simulation')
    parser.add_argument('--config', type=str, default='config/default.json',
                       help='Path to simulation configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save simulation outputs')
    parser.add_argument('--num-agents', type=int, default=10,
                       help='Number of agents in the simulation')
    parser.add_argument('--num-steps', type=int, default=1000,
                       help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                       help='Frequency of saving checkpoints')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load simulation configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_simulation(config: Dict[str, Any], num_agents: int) -> Simulation:
    """Initialize simulation with given configuration."""
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = Agent(
            agent_id=f"agent_{i}",
            config=config['agent_config'],
            embedding_dim=config['embedding_dim'],
            memory_capacity=config['memory_capacity'],
            num_concepts=config['num_concepts']
        )
        agents.append(agent)
    
    # Initialize simulation
    simulation = Simulation(
        agents=agents,
        config=config['simulation_config']
    )
    
    return simulation

def run_simulation(simulation: Simulation, config: Dict[str, Any], 
                  num_steps: int, output_dir: str, checkpoint_freq: int) -> None:
    """Run simulation for specified number of steps and save results."""
    # Setup logging and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_dir = os.path.join(output_dir, f"simulation_{timestamp}")
    os.makedirs(sim_dir, exist_ok=True)
    
    metrics_logger = MetricsLogger(sim_dir)
    network_analytics = NetworkAnalytics()
    
    # Run simulation
    for step in range(num_steps):
        # Execute one simulation step
        step_results = simulation.step()
        
        # Log metrics
        metrics_logger.log_metric("avg_reward", step_results['avg_reward'], step)
        metrics_logger.log_metric("total_interactions", step_results['num_interactions'], step)
        
        # Calculate and log network metrics
        network_metrics = network_analytics.calculate_network_metrics(
            step_results['adjacency_matrix']
        )
        for metric_name, value in network_metrics.items():
            metrics_logger.log_metric(f"network_{metric_name}", value, step)
        
        # Detect and log communities
        communities = network_analytics.detect_communities(
            step_results['adjacency_matrix']
        )
        metrics_logger.log_metric("num_communities", len(set(communities)), step)
        
        # Save checkpoint
        if (step + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(sim_dir, f"checkpoint_step_{step+1}")
            save_simulation_state({
                'step': step + 1,
                'simulation_state': simulation.get_state(),
                'config': config
            }, checkpoint_path)
            
        # Print progress
        if (step + 1) % 100 == 0:
            logging.info(f"Step {step+1}/{num_steps} completed")
    
    # Save final metrics
    metrics_logger.save_metrics()
    
    # Save final simulation state
    final_state_path = os.path.join(sim_dir, "final_state")
    save_simulation_state({
        'step': num_steps,
        'simulation_state': simulation.get_state(),
        'config': config
    }, final_state_path)

def main():
    """Main function to run the simulation."""
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run simulation
    simulation = setup_simulation(config, args.num_agents)
    
    logging.info("Starting simulation...")
    try:
        run_simulation(
            simulation=simulation,
            config=config,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
            checkpoint_freq=args.checkpoint_freq
        )
        logging.info("Simulation completed successfully")
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
