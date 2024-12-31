"""
Module: contextual_memory.py

Description:
This module manages the episodic and semantic memory components for agents. It includes 
functions for storing, retrieving, and forgetting information based on relevance and context.

Dependencies:
- numpy: For efficient numerical computations.
- torch: For tensor-based memory representation.
- simulation: For updating memory during simulation iterations.
"""

import numpy as np
import torch
from modules.simulation import Simulation
