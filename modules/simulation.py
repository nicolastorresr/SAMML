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

from modules.agent import Agent
from modules.social_adaptation import SocialAdaptation
from modules.utils import log_metrics

