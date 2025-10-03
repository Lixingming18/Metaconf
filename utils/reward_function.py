import os
import logging
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from utils.utils import parse_response, parse_json
from utils.llm import LLM_Client
from prompts.evaluate_reflection_quality import EVALUATE_REFLECTION_QUALITY_PROMPT
from utils.caculate_accuracy import calculate_accuracy

logger = logging.getLogger(__name__)

llm_client = LLM_Client(
    model_name="deepseek-v3",
    temperature=0.9,
    api_keys=[os.environ.get("DEEPSEEK_API_KEY")],
    base_url=os.environ.get("DEEPSEEK_BASE_URL"),
)


@dataclass
class RewardWeights:
    """Reward function weight configuration"""
    calibrated_reward_weight: float = 1.0
    reflection_reward_weight: float = 0.6


# Calculate calibration error reward
def calculate_calibrated_reward(correctness: float, confidence: float) -> float:
    """
    Calculate calibration error reward
    
    Args:
        correctness: Accuracy of correct answer (0-1)
        confidence: Model confidence (0-10)
    
    Returns:
        float: Calibration error reward score
    """
    if not (0 <= confidence <= 10):
        logger.warning(
            f"Invalid confidence value: {confidence}, clamping to [0, 10]"
        )
        confidence = max(0, min(10, confidence))
    
    if not (0 <= correctness <= 1):
        logger.warning(
            f"Invalid correctness value: {correctness}, clamping to [0, 1]"
        )
        correctness = max(0, min(1, correctness))
    
    # Normalize confidence to [0,1]
    confidence /= 10
    
    # Calculate calibration error: 1 - (correctness - confidence)^2
    score = 1 - (abs(correctness - confidence) ** 2)
    
    return float(score)


# Evaluate reflection quality
def evaluate_reflection_quality(
    questions: List[str], 
    model_answers: List[str], 
    self_reflections: List[str], 
    confidences: List[float], 
    accuracy_list: List[float]
) -> List[float]:
    """
    Use large model to evaluate reflection quality
    
    Args:
        questions: List of questions
        model_answers: List of model answers
        self_reflections: List of self reflections
        confidences: List of confidence scores
        accuracy_list: List of accuracy scores
    
    Returns:
        List[float]: List of reflection quality scores
    """
    item_to_process = []
    for index, (
        question,
        model_answer,
        self_reflection,
        confidence,
        accuracy,
    ) in enumerate(
        zip(questions, model_answers, self_reflections, confidences, accuracy_list)
    ):
        prompt = EVALUATE_REFLECTION_QUALITY_PROMPT.format(
            question=question,
            model_answer=model_answer,
            self_reflection=self_reflection,
            confidence_score=confidence,
            accuracy=accuracy,
        )
        item_to_process.append(
            {
                "item": {
                    "question": question,
                    "model_answer": model_answer,
                    "self_reflection": self_reflection,
                    "confidence_score": confidence,
                    "accuracy": accuracy,
                    "index": index,
                },
                "prompt": prompt,
            }
        )
    
    try:
        results = llm_client.process_batch(item_to_process)
    except Exception as e:
        logger.error(f"Error processing reflection quality evaluation: {e}")
        return [0.0] * len(questions)

    for result in results:
        try:
            reflection_quality = result["model_answer_list"][0]
            result.pop("model_answer_list")
            reflection_quality = parse_json(reflection_quality)
            reflection_quality = reflection_quality["total_score"] / 2
        except Exception as e:
            logger.error(
                f"Error parsing reflection quality result: {traceback.format_exc()}"
            )
            reflection_quality = 0
        result["reflection_score"] = reflection_quality

    sorted_results = sorted(results, key=lambda x: x["index"])
    reflection_scores = [result["reflection_score"] for result in sorted_results]
    return reflection_scores


def calculate_reward(
    prompts: List[List[Dict[str, Any]]], 
    completions: List[List[Dict[str, Any]]], 
    answer: List[str], 
    weights: Optional[RewardWeights] = None,
    **kwargs
) -> List[float]:
    """
    Calculate comprehensive reward scores
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers
        weights: Reward weight configuration, use default weights if None
        **kwargs: Other parameters
    
    Returns:
        List[float]: List of reward scores
    """
    if weights is None:
        weights = RewardWeights()
    
    # Extract model responses
    model_responses = [completion[0]["content"] for completion in completions]

    # Parse task information from prompts
    task_types, question_titles, questions = [], [], []
    for prompt in prompts:
        for conversation in prompt:
            if conversation["role"] == "user":
                task_types.append(conversation["task_type"])
                question_titles.append(conversation.get("title", ""))
                questions.append(conversation["content"])

    correct_answers = answer

    # Parse model responses
    answer_list, reflection_list, confidence_list = [], [], []
    formatted_rewards_list = []

    for model_response in model_responses:
        try:
            answer, reflection, confidence = parse_response(model_response)
            temp_reward = 0
            if reflection:
                temp_reward += 1
            if confidence:
                temp_reward += 1
            formatted_rewards_list.append(temp_reward / 2)
            answer_list.append(answer)
            reflection_list.append(reflection)
            confidence_list.append(confidence)
        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            formatted_rewards_list.append(0)
            answer_list.append("")
            reflection_list.append("")
            confidence_list.append(0)

    # Calculate accuracy
    try:
        accuracy_list = calculate_accuracy(
            questions, correct_answers, model_responses, task_types, question_titles
        )
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        accuracy_list = [0.0] * len(questions)
    
    logger.info(f"Confidence list: {confidence_list}")
    
    # Calculate calibration error reward
    calibrated_reward_list = [
        calculate_calibrated_reward(accuracy, confidence)
        for accuracy, confidence in zip(accuracy_list, confidence_list)
    ]

    # Calculate reflection quality reward
    try:
        reflection_rewards_list = evaluate_reflection_quality(
            questions, answer_list, reflection_list, confidence_list, accuracy_list
        )
    except Exception as e:
        logger.error(f"Error evaluating reflection quality: {e}")
        reflection_rewards_list = [0.0] * len(questions)

    # Adjust other rewards based on formatted reward
    for index, (calibrated_reward, reflection_reward) in enumerate(
        zip(calibrated_reward_list, reflection_rewards_list)
    ):
        if formatted_rewards_list[index] == 0:
            calibrated_reward_list[index] = 0
            reflection_rewards_list[index] = 0


    # Calculate comprehensive reward
    total_rewards = []
    for calibrated_reward, reflection_reward in zip(
        calibrated_reward_list,
        reflection_rewards_list
    ):
        reward = (
            weights.calibrated_reward_weight * calibrated_reward
            + weights.reflection_reward_weight * reflection_reward
        )
        
        logger.info(
            f"Reward breakdown - Calibrated: {calibrated_reward:.3f}, "
            f"Reflection: {reflection_reward:.3f}, "
            f"Final: {reward:.3f}"
        )
        total_rewards.append(reward)
    
    logger.info(f"Total rewards: {total_rewards}")
    logger.info("#" * 100)
    return total_rewards

