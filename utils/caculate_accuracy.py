import os
import re
from typing import List, Dict, Any
from collections import defaultdict


from utils.llm import LLM_Client
from utils.fact_score import FactualityScorer
from prompts.math_answer_check import MATH_ANSWER_CHECK_PROMPT


fact_scorer = FactualityScorer()

llm_client = LLM_Client(
    model_name="deepseek-v3",
    temperature=0.9,
    api_keys=[os.environ.get("DEEPSEEK_API_KEY")],
    base_url=os.environ.get("DEEPSEEK_BASE_URL"),
)


# Calculate answer accuracy
def calculate_accuracy(
    questions, correct_answers, model_answers, task_types, question_title_list=List[str]
):
    """
    Use different methods to calculate accuracy based on question type
    """

    print(f"task_types is {task_types}")

    grouped_data = defaultdict(list)
    for i, task_type in enumerate(task_types):
        grouped_data[task_type].append(
            {
                "question": questions[i],
                "correct_answer": correct_answers[i],
                "model_answer": model_answers[i],
                "question_title": question_title_list[i],
                "index": i,
            }
        )

    accuracy_list = []
    if task_type == "math":
        accuracy_list.extend(calculate_accuracy_for_math(grouped_data[task_type]))
    elif task_type == "long_fact":
        accuracy_list.extend(
            calculate_accuracy_for_long_fact(grouped_data[task_type], fact_scorer)
        )

    # Re-sort accuracy based on index
    accuracy_list = sorted(accuracy_list, key=lambda x: x["index"])

    # Return only scores
    accuracy_list = [item["accuracy"] for item in accuracy_list]

    print(f"accuracy_list is {accuracy_list}")
    return accuracy_list


def calculate_accuracy_for_math(grouped_data):
    """
    Calculate accuracy for math problems
    """

    def parse_math_answer(items: List[Dict[str, Any]]):
        """
        Parse math problem analysis answers
        """
        accuracy_list = []
        for item in items:
            correct_analysis = item["model_answer_list"]
            index = item["index"]
            if not correct_analysis:
                accuracy = 0
                accuracy_list.append({"index": index, "accuracy": accuracy})
                continue
            correct_analysis = correct_analysis[0]
            if "Final Score:" not in correct_analysis:
                accuracy = 0
                accuracy_list.append({"index": index, "accuracy": accuracy})
                continue
            sentences = correct_analysis.split("\n")
            sentence_with_score = sentences[-1].replace("#", "").replace("*", "")
            if re.findall(r"(\d+\.\d+|\d+)", sentence_with_score):
                score = re.findall(r"(\d+\.\d+|\d+)", sentence_with_score)[0]
                if 0 < float(score) <= 1.0:
                    accuracy = float(score)
                else:
                    accuracy = 0
            else:
                accuracy = 0
            accuracy_list.append({"index": index, "accuracy": accuracy})
        return accuracy_list

    item_to_process = []
    for item in grouped_data:
        prompt = MATH_ANSWER_CHECK_PROMPT.format(
            question=item["question"],
            correct_answer=item["correct_answer"],
            model_answer=item["model_answer"],
        )
        item_to_process.append({"item": item, "prompt": prompt})
    results = llm_client.process_batch(item_to_process)

    accuracy_list = parse_math_answer(results)
    return accuracy_list


def calculate_accuracy_for_long_fact(
    grouped_data: List[Dict[str, Any]], fact_scorer: FactualityScorer
) -> List[float]:
    """
    Calculate accuracy for long text questions
    """
    fact_score_list = fact_scorer.factuality_count_reward_func(
        prompts=[item["question"] for item in grouped_data],
        completions=[item["model_answer"] for item in grouped_data],
        title=[item["question_title"] for item in grouped_data],
    )
    accuracy_list = [
        {"index": idx, "accuracy": fact_score}
        for idx, fact_score in enumerate(fact_score_list)
    ]
    return accuracy_list
