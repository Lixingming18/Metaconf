import re
import math
import os
import json
import yaml
import wandb
import random
import logging
import traceback
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from unsloth import FastLanguageModel, PatchFastRL
from peft import (
    AutoPeftModelForCausalLM,
    get_peft_model,
    PeftModel,
    PeftConfig,
    LoraConfig,
    TaskType,
)
from accelerate import Accelerator
from tqdm import tqdm

# Patch FastRL for GRPO
PatchFastRL("GRPO", FastLanguageModel)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reward_function import calculate_reward, formatted_reward, RewardWeights
from utils.utils import (
    initialize_wandb_from_yaml,
    get_reward_output_dir,
    get_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "math"
    sample_size: int = 1000
    seed: int = 42
    file_path: Optional[str] = None



@dataclass
class RewardConfig:
    """Reward function configuration"""
    calibrated_reward_weight: float = 1.0
    reflection_reward_weight: float = 0.6
    accuracy_reward_weight: float = 0.2
    formatted_reward_weight: float = 0.0
    
    def to_weights(self) -> RewardWeights:
        """Convert to RewardWeights object"""
        return RewardWeights(
            calibrated_reward_weight=self.calibrated_reward_weight,
            reflection_reward_weight=self.reflection_reward_weight,
            accuracy_reward_weight=self.accuracy_reward_weight,
        )


class DatasetManager:
    """Dataset manager"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset"""
        self.logger.info(f"Preparing dataset: {self.config.name}")
        
        if self.config.name == "math":
            return self._load_math_dataset()
        elif self.config.name == "long_form_qa":
            return self._load_long_form_qa_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.name}")
    
    def _load_math_dataset(self) -> Dataset:
        """Load math dataset"""
        file_path = self.config.file_path or "../datasets/stage2/math.json"
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
            
            # Random sampling
            random.seed(self.config.seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:self.config.sample_size])
            
            self.logger.info(f"Loaded math dataset with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading math dataset: {e}")
            raise
    
    def _load_long_form_qa_dataset(self) -> Dataset:
        """Load long form QA dataset"""
        file_path = self.config.file_path or "../datasets/stage2/long_form_qa.json"
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
            
            # Random sampling
            random.seed(self.config.seed)
            indices = list(range(len(dataset)))
            dataset = dataset.select(indices[:self.config.sample_size])
            
            self.logger.info(f"Loaded long_form_qa dataset with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading long_form_qa dataset: {e}")
            raise


class ModelManager:
    """Model manager"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        try:
            # Load model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name_or_path,
                fast_inference=True,
                load_in_4bit=False,
                max_lora_rank=self.config.lora_r,
                max_seq_length=self.config.max_seq_length,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                attn_implementation=self.config.attn_implementation,
            )
            
            # Configure PEFT model
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                use_gradient_checkpointing="unsloth",
                random_state=42,  # Can be obtained from config
            )
            
            # Ensure tokenizer has padding token
            self._setup_tokenizer()
            
            self.logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_tokenizer(self):
        """Set up tokenizer padding token"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info(f"Set pad_token to {self.tokenizer.eos_token}")
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"Set pad_token_id to {self.tokenizer.eos_token_id}")


class GRPOTrainerManager:
    """GRPO trainer manager"""
    
    def __init__(self, model: Any, tokenizer: Any, training_args: GRPOConfig, 
                 reward_config: RewardConfig, dataset: Dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.reward_config = reward_config
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)
        self.trainer = None
    
    def create_trainer(self, callbacks: Optional[List] = None) -> GRPOTrainer:
        """Create GRPO trainer"""
        self.logger.info("Creating GRPO trainer")
        
        # Create reward function wrapper
        def reward_wrapper(prompts, completions, answer, **kwargs):
            return calculate_reward(
                prompts, completions, answer, 
                weights=self.reward_config.to_weights(),
                **kwargs
            )
        
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[reward_wrapper],
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=None,
            callbacks=callbacks,
        )
        
        self.logger.info("GRPO trainer created successfully")
        return self.trainer
    
    def train(self) -> Dict[str, Any]:
        """Execute training"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")
        
        self.logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
            f"for {self.training_args.num_train_epochs} epochs ***"
        )
        
        try:
            # Train model
            train_result = self.trainer.train(
                resume_from_checkpoint=(
                    get_checkpoint(self.training_args)
                    if self.training_args.resume_from_checkpoint
                    else None
                )
            )
            
            # Save metrics
            metrics = train_result.metrics
            metrics["train_samples"] = len(self.dataset)
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            self.logger.info("*** Training completed ***")
            return train_result
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def save_model(self):
        """Save model and tokenizer"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        self.logger.info("*** Saving model ***")
        
        try:
            self.trainer.model.config.use_cache = True
            self.model.save_pretrained(self.training_args.output_dir)
            self.logger.info(f"Model saved to {self.training_args.output_dir}")
            
            self.tokenizer.save_pretrained(self.training_args.output_dir)
            self.logger.info(f"Tokenizer saved to {self.training_args.output_dir}")
            
            self.logger.info("*** Training process completed! ***")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise


class GRPOTrainingPipeline:
    """GRPO training pipeline"""
    
    def __init__(self, model_config: GRPOConfig, 
                 dataset_config: DatasetConfig,
                 reward_config: RewardConfig):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.reward_config = reward_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dataset_manager = DatasetManager(dataset_config)
        self.model_manager = ModelManager(model_config)
        self.trainer_manager = None
    
    def run(self, training_args: GRPOConfig, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Run complete training pipeline"""
        self.logger.info("Starting GRPO training pipeline")
        
        try:
            dataset = self.dataset_manager.prepare_dataset()
            self.logger.info(f"Dataset prepared with {len(dataset)} samples")
            
            model, tokenizer = self.model_manager.load_model_and_tokenizer()
            
            self.trainer_manager = GRPOTrainerManager(
                model, tokenizer, training_args, self.reward_config, dataset
            )
            
            trainer = self.trainer_manager.create_trainer(callbacks)
            
            train_result = self.trainer_manager.train()
            
            self.trainer_manager.save_model()
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {e}")
            raise


def load_config_from_yaml(config_path: str) -> Tuple[DatasetConfig, RewardConfig]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract configuration
        dataset_config = DatasetConfig(**config_data.get('dataset', {}))
        reward_config = RewardConfig(**config_data.get('reward', {}))
        
        return dataset_config, reward_config
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def main():
    """Main function"""
    parser = TrlParser((ModelConfig, GRPOConfig))
    model_args, training_args = parser.parse_args_and_config()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    custom_config_file = getattr(training_args, 'custom_config_file', "./configs/custom_config.yaml")
    
    wandb_instance = initialize_wandb_from_yaml(custom_config_file)
    if wandb_instance:
        logger.info("WandB is set up. You can now use wandb.log() in the training loop.")
    else:
        logger.info("WandB initialization failed or not enabled.")
    
    try:
        dataset_config, reward_config = load_config_from_yaml(custom_config_file)
        
        pipeline = GRPOTrainingPipeline(model_args, dataset_config, reward_config)
        
        callbacks = None
        result = pipeline.run(training_args, callbacks=callbacks)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
