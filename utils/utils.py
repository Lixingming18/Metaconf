import os
import json
import logging
import yaml
import wandb
from transformers import GRPOConfig, get_last_checkpoint
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
import os
import re

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
GRPO_path = os.path.join(current_dir, "config.yaml")


def load_yaml_config(yaml_path=GRPO_path):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {yaml_path}")
        return config
    except Exception as e:
        logger.error(
            f"Failed to load config file {yaml_path}: {str(e)}, cannot continue"
        )
        sys.exit(1)


_timestamp_singleton = None


def get_output_timestamp():
    global _timestamp_singleton
    if _timestamp_singleton is None:
        _timestamp_singleton = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _timestamp_singleton


def get_reward_output_dir():
    yaml_config = load_yaml_config()
    project_name = yaml_config.get("wandb_project", "default_project")

    timestamp = get_output_timestamp()
    output_dir = os.path.join("reward_outputs", project_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created reward function output directory: {output_dir}")
    return output_dir


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json_output(data: Dict, function_name: str, sample_index: int = None):
    rewards_output_dir = get_reward_output_dir()

    if sample_index is not None:
        filename = f"{function_name}_sample_{sample_index}.json"
    else:
        filename = f"{function_name}.json"

    filepath = os.path.join(rewards_output_dir, filename)

    existing_data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Cannot parse existing JSON file: {filepath}, will create new file"
            )

    # Merge data
    if isinstance(data, dict) and isinstance(existing_data, dict):
        for key, value in data.items():
            if (
                key in existing_data
                and isinstance(value, list)
                and isinstance(existing_data[key], list)
            ):
                existing_data[key].extend(value)
            else:
                existing_data[key] = value
        merged_data = existing_data
    else:
        merged_data = data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    return filepath


# Parse response, return answer, reflection, confidence
def parse_response(response, args=None):

    response = re.sub(r"^.*?assistant\n\n", "", response, flags=re.DOTALL)
    answer_pattern = r"Answer:\s*(.*?)\s*Self-reflection:"
    reflection_pattern = r"Self-reflection:\s*(.*?)\s*Confidence:"
    confidence_pattern = r"Confidence:\s*(\d+\.\d+|\d+)\s*$"

    answer_match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)
    reflection_match = re.search(
        reflection_pattern, response, re.DOTALL | re.IGNORECASE
    )
    confidence_match = re.search(
        confidence_pattern, response, re.DOTALL | re.IGNORECASE
    )

    # Extract and clean results
    answer = answer_match.group(1).strip() if answer_match else response
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1).strip())
        except Exception as e:
            logger.error(f"Error parsing confidence: {e}")
            confidence = 0
    else:
        confidence = 0
    if confidence < 0 or confidence > 10:
        confidence = 0

    return answer, reflection, confidence


def parse_json(json_text):
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, json_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    return json.loads(json_text)


def normalize_answer(answer: List[str]):
    return [a.strip().lower() for a in answer]


def initialize_wandb_from_yaml(yaml_file_path: str):
    """Initialize WandB from YAML configuration file"""
    try:
        if not os.path.isabs(yaml_file_path):
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            full_yaml_path = os.path.join(current_script_dir, yaml_file_path)
        else:
            full_yaml_path = yaml_file_path

        if not os.path.exists(full_yaml_path):
            logger.error(f"YAML config file '{full_yaml_path}' not found.")
            return None

        with open(full_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        report_to = config.get("report_to")
        if report_to != "wandb":
            logger.info(
                f"WandB reporting not enabled in YAML (report_to: {report_to}). Skipping WandB initialization."
            )
            return None

        wandb_project = config.get("wandb_project")
        wandb_entity = config.get("wandb_entity")
        run_name = config.get("run_name")

        if not all([wandb_project, wandb_entity, run_name]):
            logger.error("Missing WandB parameters in YAML file.")
            logger.error(f"  - wandb_project: {wandb_project}")
            logger.error(f"  - wandb_entity: {wandb_entity}")
            logger.error(f"  - run_name: {run_name}")
            return None

        wandb.init(
            project=wandb_project, entity=wandb_entity, name=run_name, config=config
        )

        logger.info(f"WandB successfully initialized!")
        logger.info(f"  Project: {wandb_project}")
        logger.info(f"  Entity: {wandb_entity}")
        logger.info(f"  Run Name: {run_name}")
        logger.info(f"  URL: {wandb.run.get_url()}")

        return wandb

    except FileNotFoundError:
        logger.error(f"YAML config file '{full_yaml_path}' not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file '{full_yaml_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unknown error initializing WandB: {e}")
        return None


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def load_yaml_config(yaml_path="./grpo_train_config.yaml"):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {yaml_path}")
        return config
    except Exception as e:
        logger.error(
            f"Failed to load config file {yaml_path}: {str(e)}, cannot continue"
        )
        sys.exit(1)


def get_reward_output_dir():
    yaml_config = load_yaml_config()
    project_name = yaml_config.get("wandb_project", "default_project")

    timestamp = get_output_timestamp()
    output_dir = os.path.join("reward_outputs", project_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created reward function output directory: {output_dir}")
    return output_dir
