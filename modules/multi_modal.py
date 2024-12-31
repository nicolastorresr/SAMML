"""
Module: multi_modal.py

Description:
This module implements the processing of multi-modal inputs (text, images, and audio) 
using pre-trained models. It includes pipelines for feature extraction and the integration 
of multiple signals into a unified representation for agents to process.

Dependencies:
- transformers: For processing text using pre-trained language models.
- torchvision: For processing image data.
- numpy: For numerical operations on extracted features.
- utils: For loading and pre-processing raw data inputs.
"""

from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from .utils import DataLoader, setup_device

@dataclass
class ModalityConfig:
    """Configuration for each modality"""
    enabled: bool
    model_name: str
    input_dim: int
    output_dim: int

class FeatureExtractor:
    """Base feature extractor for different modalities"""
    def __init__(self, device: torch.device):
        self.device = device

    def extract(self, input_data: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement extract()")

class TextFeatureExtractor(FeatureExtractor):
    """Feature extractor for text using transformer models"""
    def __init__(self, model_name: str, device: torch.device):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, text: str) -> torch.Tensor:
        """
        Extract features from input text using a pre-trained transformer model
        Args:
            text: Input text to process
        Returns:
            torch.Tensor: Text features
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Use [CLS] token representation as text features
            return outputs.last_hidden_state[:, 0, :]

class ImageFeatureExtractor(FeatureExtractor):
    """Feature extractor for images using pre-trained models"""
    def __init__(self, model_name: str = "resnet50", device: torch.device = None):
        super().__init__(device)
        # Load pre-trained model
        self.model = getattr(models, model_name)(pretrained=True).to(device)
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image using a pre-trained CNN
        Args:
            image: Input image tensor
        Returns:
            torch.Tensor: Image features
        """
        with torch.no_grad():
            image = self.transform(image).to(self.device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = self.model(image)
            return features.squeeze()

class MultiModalFusion(nn.Module):
    """Module for fusing features from different modalities"""
    def __init__(self, modality_configs: Dict[str, ModalityConfig]):
        super().__init__()
        
        self.configs = modality_configs
        total_input_dim = sum(config.input_dim for config in modality_configs.values() 
                            if config.enabled)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from different modalities
        Args:
            features: Dictionary containing features from each modality
        Returns:
            torch.Tensor: Fused multi-modal representation
        """
        # Concatenate features from enabled modalities
        concat_features = []
        for modality, config in self.configs.items():
            if config.enabled and modality in features:
                concat_features.append(features[modality])
        
        if not concat_features:
            raise ValueError("No features provided for enabled modalities")
        
        x = torch.cat(concat_features, dim=-1)
        return self.fusion_layers(x)

class MultiModalProcessor:
    """Main class for multi-modal processing"""
    def __init__(self, config: Dict[str, ModalityConfig]):
        """
        Initialize the multi-modal processor
        Args:
            config: Configuration dictionary for each modality
        """
        self.device = setup_device()
        self.config = config
        
        # Initialize extractors for enabled modalities
        self.extractors = {}
        if config.get('text', ModalityConfig(False, '', 0, 0)).enabled:
            self.extractors['text'] = TextFeatureExtractor(
                config['text'].model_name, self.device)
        if config.get('image', ModalityConfig(False, '', 0, 0)).enabled:
            self.extractors['image'] = ImageFeatureExtractor(
                config['image'].model_name, self.device)
        
        # Initialize fusion module
        self.fusion_module = MultiModalFusion(config).to(self.device)

    def process(self, inputs: Dict[str, Union[str, torch.Tensor]]) -> torch.Tensor:
        """
        Process multi-modal inputs and return a unified representation
        Args:
            inputs: Dictionary containing inputs for each modality
        Returns:
            torch.Tensor: Unified multi-modal representation
        """
        features = {}
        
        for modality, extractor in self.extractors.items():
            if modality in inputs and self.config[modality].enabled:
                try:
                    features[modality] = extractor.extract(inputs[modality])
                except Exception as e:
                    logging.error(f"Error processing {modality} modality: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("Could not extract features from any modality")
        
        return self.fusion_module(features)

    def save_state(self, path: str):
        """
        Save the state of the multi-modal processor
        Args:
            path: Path to save the state
        """
        state = {
            'config': self.config,
            'fusion_state': self.fusion_module.state_dict()
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """
        Load the state of the multi-modal processor
        Args:
            path: Path to load the state from
        """
        state = torch.load(path)
        self.config = state['config']
        self.fusion_module.load_state_dict(state['fusion_state'])
