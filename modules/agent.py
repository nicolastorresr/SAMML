"""
Module: agent.py

Description:
This module defines the base class for agents in the simulation. Each agent can process 
multi-modal inputs, interact with other agents, and update its state based on feedback 
from the social adaptation module.

Dependencies:
- multi_modal: For processing multi-modal signals.
- social_adaptation: For updating agent behavior based on interactions.
- contextual_memory: For managing agent memory.
"""

from modules.multi_modal import process_text, process_image
from modules.social_adaptation import SocialAdaptation
from modules.contextual_memory import EpisodicMemory, SemanticMemory
