import os
import sys
import json
import time
import random
import logging
import threading
from queue import Queue
from typing import List, Dict, Any, Set
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import httpx
from openai import APIError
from tqdm import tqdm
from random import shuffle
import traceback
import fcntl  # For file locking


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

http_client = httpx.Client(verify=False)


class LLM_Client:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.9,
        api_keys: List[str] = None,
        base_url: str = "",
        max_retries: int = 3,
        num_threads: int = 5,
        max_tokens: int = 1000,
        per_prompt_answer_times: int = 1,
    ):
        self.model_name = model_name
        self.api_keys = api_keys
        self.base_url = base_url
        self.max_retries = max_retries
        self.num_threads = num_threads
        self.max_tokens = max_tokens
        self.per_prompt_answer_times = per_prompt_answer_times
        self.temperature = temperature
        self.api_key_queue = Queue()
        for key in api_keys:
            self.api_key_queue.put(key)
        self.api_key_lock = threading.Lock()

        self.clients = {}
        for key in api_keys:
            self.clients[key] = OpenAI(
                api_key=key, base_url=base_url, http_client=http_client
            )

        self.results_queue = Queue()
        self.file_lock = threading.Lock()
        self.output_file = None

    def get_api_key(self) -> str:
        """Thread-safe method to get an API key"""

        with self.api_key_lock:
            key = self.api_key_queue.get()
            self.api_key_queue.put(key)
            return key

    def save_result_immediately(self, result: Dict[str, Any], output_file: str):
        """Save a single result immediately to file"""
        with self.file_lock:
            try:
                with open(output_file, "a", encoding="utf-8") as f:
                    # Use file locking to ensure thread safety
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()  # Ensure data is written to disk
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logging.error(f"Error saving result: {e}")

    def load_processed_questions(self, output_file: str) -> Set[str]:
        """Load already processed questions from output file to support resume"""
        processed_questions = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            # Use question as unique identifier
                            question_id = data.get("question_id") or data.get(
                                "question", ""
                            )
                            if question_id:
                                processed_questions.add(question_id)
                logging.info(
                    f"Found {len(processed_questions)} already processed questions"
                )
            except Exception as e:
                logging.error(f"Error loading processed questions: {e}")
        return processed_questions

    def call_api_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI API with retry mechanism, including original item fields in the result."""

        original_item = payload.get("item", {})
        prompt = payload["prompt"]
        output_file = payload.get("output_file", "")
        model_answer_list = []

        # Generate multiple answers for the same question
        for i in range(self.per_prompt_answer_times):
            success = False
            for attempt in range(self.max_retries):
                try:
                    api_key = self.get_api_key()
                    client = self.clients[api_key]

                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    model_answer = response.choices[0].message.content
                    model_answer_list.append(model_answer)
                    success = True
                    break
                except APIError as e:
                    logging.error(
                        f"API error on attempt {attempt + 1}: {traceback.format_exc()}"
                    )
                    if attempt == self.max_retries - 1:
                        logging.error(
                            f"Failed to get answer {i + 1} after {self.max_retries} attempts"
                        )
                    time.sleep(1)
                    continue

                except Exception as e:
                    logging.error(
                        f"Unexpected error on attempt {attempt + 1}: {traceback.format_exc()}"
                    )
                    if attempt == self.max_retries - 1:
                        logging.error(
                            f"Failed to get answer {i + 1} after {self.max_retries} attempts"
                        )
                    time.sleep(1)
                    continue

            # If we couldn't get this answer after all retries, we still continue to try the next one
            if not success:
                logging.warning(
                    f"Could not generate answer {i + 1} for question: {original_item.get('question', 'Unknown')[:50]}..."
                )

        result = original_item.copy()
        result["model_answer_list"] = model_answer_list
        # Save result immediately

        if output_file:
            self.save_result_immediately(result, output_file)

        return result

    def process_batch(
        self, items_with_prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of items (original_item + prompt) using thread pool with real-time progress updates"""
        results = []
        progress_bar = tqdm(total=len(items_with_prompts), desc="Processing questions")

        def process_item(item):
            try:
                result = self.call_api_with_retry(item)
                progress_bar.update(1)
                return result
            except Exception as e:
                logging.error(f"Error processing item: {traceback.format_exc()}")
                progress_bar.update(1)
                return None

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(process_item, item) for item in items_with_prompts
            ]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)

        progress_bar.close()
        return results

    def create_dataset(
        self, items_to_process: List[Dict[str, Any]], output_file: str = "dataset.jsonl"
    ):
        """Create dataset from items_to_process and save to file with resume capability"""

        # Ensure output file has .jsonl extension for line-by-line saving
        if not output_file.endswith(".jsonl"):
            output_file = output_file.replace(".json", ".jsonl")

        logging.info(f"Output file: {output_file}")

        # Load already processed questions
        processed_questions = self.load_processed_questions(output_file)

        # Filter out already processed items
        remaining_items = []
        for item in items_to_process:
            question_id = item.get("item", {}).get("question", "")
            if question_id not in processed_questions:
                remaining_items.append(item)

        logging.info(f"Total items: {len(items_to_process)}")
        logging.info(f"Already processed: {len(processed_questions)}")
        logging.info(f"Remaining to process: {len(remaining_items)}")

        if not remaining_items:
            logging.info("All items have been processed!")

        # Prepare prompts for remaining items only
        remaining_items_with_prompts = []
        for item_data in remaining_items:
            if (
                isinstance(item_data, dict)
                and "item" in item_data
                and "prompt" in item_data
            ):
                # Already prepared
                item_data["output_file"] = output_file
                remaining_items_with_prompts.append(item_data)
            else:
                # Need to prepare
                remaining_items_with_prompts.append(
                    {
                        "item": item_data,
                        "prompt": item_data.get("prompt", ""),
                        "output_file": output_file,
                    }
                )

        logging.info(
            f"Starting to process {len(remaining_items_with_prompts)} remaining items..."
        )

        self.process_batch(remaining_items_with_prompts)

        logging.info(f"Processing complete! Results saved to {output_file}")

    def prepare_prompts(
        self, dataset_items: List[Dict], few_shot_examples: str = ""
    ) -> List[Dict[str, Any]]:
        """Prepare prompts for each dataset item"""
        items_with_prompts = []

        for item in dataset_items:
            question = item.get("question", "")

            if few_shot_examples:
                prompt = f"{few_shot_examples}{question}\nAnswer:"
            else:
                prompt = question

            items_with_prompts.append({"item": item, "prompt": prompt})

        return items_with_prompts


if __name__ == "__main__":

    pass
