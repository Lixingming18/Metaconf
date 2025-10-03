import os
import re
import sys
import logging
import traceback
from typing import List, Optional
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from utils.FActScore.factscore.factscorer import FactScorer
from utils.utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)


class FactualityScorer:
    def __init__(self):
        self.FACT_SCORER = None
        self._initialize_fact_scorer()

    def _initialize_fact_scorer(self):
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "")

            db_path = os.environ.get("FACTSCORE_DB_PATH", "")
            if not os.path.exists(db_path):
                logger.error(f"Knowledge base file does not exist: {db_path}")
                return None

            logger.info("Initializing FactScorer using standard method")

            fs = FactScorer(
                openai_key=openai_api_key,
                base_url=base_url,
                cache_dir=None,
                af_model_version="deepseek-v3",
                use_nli=True,
                nli_model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                nli_entailment_threshold=0.3,
                verbose=True,
            )
            fs.register_knowledge_source(db_path=db_path)

            self.FACT_SCORER = fs
            logger.info(
                f"Successfully initialized FactScorer and registered knowledge source: {db_path}"
            )
            return self.FACT_SCORER

        except Exception as e:
            logger.error(f"Failed to initialize FactScorer: {str(e)}")
            return None

    def get_fact_scorer(self):
        if self.FACT_SCORER is not None:
            return self.FACT_SCORER

        return self._initialize_fact_scorer()

    def factuality_count_reward_func(
        self,
        prompts: List[str],
        completions: List[str],
        title: Optional[List[str]] = None,
        index: int = 0,
        use_total_count: bool = False,
        **kwargs,
    ) -> List[float]:
        rewards = [0.0] * len(completions)
        log_entries = []

        final_results = []

        fs = self.get_fact_scorer()

        if fs is None:
            logger.error("Cannot get FactScorer instance, returning 0 for all samples")

            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                log_entries.append(
                    {
                        "sample_index": i,
                        "prompt": prompt,
                        "completion": completion,
                        "error": "FactScorer initialization failed",
                        "reward": 0.0,
                        "timestamp": get_timestamp(),
                    }
                )

            save_json_output({"factuality_results": log_entries}, "factuality_reward")
            return rewards

        try:
            topics = []
            answers = []
            valid_indices = []

            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                try:

                    answer_text = completion.strip()

                    if title and i < len(title) and title[i]:
                        topic = title[i]
                    else:
                        topic = prompt.strip()

                    topics.append(topic)
                    answers.append(answer_text)
                    valid_indices.append(i)

                except Exception as e:
                    logger.error(f"Error processing example {i}: {str(e)}")
                    log_entries.append(
                        {
                            "sample_index": i,
                            "prompt": prompt,
                            "completion": completion,
                            "error": str(e),
                            "reward": 0.0,
                            "timestamp": get_timestamp(),
                        }
                    )

            if not valid_indices:
                logger.warning("No valid answers found")
                save_json_output(
                    {"factuality_results": log_entries}, "factuality_reward"
                )
                return rewards
            # Batch evaluate factual accuracy using atomic fact count scoring mode
            fs_results = fs.get_score(
                topics=topics,
                generations=answers,
                gamma=10,
                use_nli=True,
                use_async_af_generation=True,
                count_supported=True,
            )

            if isinstance(fs_results, list):
                for idx, i in enumerate(valid_indices):
                    if idx < len(fs_results):
                        result = fs_results[idx]

                        supported_facts_count = float(result.get("score", 0))
                        logger.info(f"supported_facts_count: {supported_facts_count}")
                        factuality_score = min(supported_facts_count / 15.0, 1.0)

                        log_entry = {
                            "sample_index": i,
                            "topic": topics[idx],
                            "prompt": prompts[i],
                            "answer": answers[idx],
                            "supported_facts_count": float(supported_facts_count),
                            "timestamp": get_timestamp(),
                        }

                        if (
                            "decisions" in result
                            and result["decisions"]
                            and result["decisions"][0]
                        ):
                            atomic_facts = []
                            supported_count = 0
                            total_count = 0

                            for decision in result["decisions"][0]:
                                is_supported = bool(decision["is_supported"])
                                atomic_facts.append(
                                    {
                                        "fact": decision["atom"],
                                        "supported": is_supported,
                                    }
                                )
                                total_count += 1
                                if is_supported:
                                    supported_count += 1

                            log_entry["atomic_facts"] = atomic_facts
                            log_entry["supported_facts_count"] = (
                                f"{supported_count}/{total_count}"
                            )
                            log_entry["supported_facts_ratio"] = float(
                                supported_count / total_count if total_count > 0 else 0
                            )
                            if use_total_count:
                                log_entry["factuality_score"] = float(
                                    supported_count / total_count
                                )
                            else:
                                log_entry["factuality_score"] = float(
                                    supported_count / 20
                                )
                            log_entry["factuality_score"] = max(
                                min(log_entry["factuality_score"], 1.0), 0.0
                            )
                            temp = {
                                "topic": log_entry["topic"],
                                "prompt": log_entry["prompt"],
                                "answer": log_entry["answer"],
                                "atomic_facts": atomic_facts,
                                "supported_facts_count": f"{supported_count}/{total_count}",
                                "factuality_score": round(
                                    log_entry["factuality_score"], 2
                                ),
                            }
                            rewards[i] = log_entry["factuality_score"]
                            final_results.append(temp)
                        else:
                            # When no decisions information is available, create a default temp object
                            temp = {
                                "topic": log_entry["topic"],
                                "prompt": log_entry["prompt"],
                                "answer": log_entry["answer"],
                                "atomic_facts": [],
                                "supported_facts_count": "0/0",
                                "factuality_score": 0.0,
                            }
                            rewards[i] = 0.0
                            final_results.append(temp)
                        log_entries.append(log_entry)

                fs_summary = {
                    "overall_score": float(
                        np.mean([r.get("score", 0) for r in fs_results])
                    ),
                    "respond_ratio": float(
                        fs_results[0].get("respond_ratio", 1.0) if fs_results else 0
                    ),
                    "num_facts_per_response": float(
                        np.mean(
                            [r.get("num_facts_per_response", 0) for r in fs_results]
                        )
                    ),
                }
            else:
                fs_summary = {
                    "overall_score": float(fs_results.get("score", 0)),
                    "respond_ratio": float(fs_results.get("respond_ratio", 0)),
                    "num_facts_per_response": float(
                        fs_results.get("num_facts_per_response", 0)
                    ),
                }

            all_results = {"summary": fs_summary, "detailed_results": log_entries}
            save_json_output(final_results, f"factuality_reward_final_{index}")

            save_json_output(all_results, f"factuality_reward_{index}")

        except Exception as e:
            logger.error(f"Factuality evaluation error: {str(e)}")
            error_log = {
                "error": str(e),
                "traceback": (
                    traceback.format_exc() if "traceback" in sys.modules else None
                ),
            }
            save_json_output(error_log, "factuality_error")

        return rewards


if __name__ == "__main__":
    scorer = FactualityScorer()
    prompts = ["Who is Nathan Wolfe?", "Who is Nathan Wolfe?"]
    completions = [
        "Nathan Wolfe is a medical anthropologist and entrepreneur. He is best known for being the founder of Metabiota, a company that uses data and analytics to understand and predict the spread of infectious diseases. Wolfe has also been involved in various other ventures, including the Global Viral Forecast Initiative, which aims to track and predict the emergence of new viral diseases.\n\nWolfe has been recognized for his work in the field of infectious disease research, including being named one of Time Magazine's 100 most influential people in the world in 2009. He has also been featured in various media outlets, including The New York Times, The Wall Street Journal, and NPR, for his work on predicting and preventing the spread of infectious diseases.",
        "Nathan Wolfe is a medical anthropologist and entrepreneur. He is best known for being the founder of Metabiota, a company that uses data and analytics to understand and predict the spread of infectious diseases. Wolfe has also been involved in various other ventures, including the Global Viral Forecast Initiative, which aims to track and predict the emergence of new viral diseases.\n\nWolfe has been recognized for his work in the field of infectious disease research, including being named one of Time Magazine's 100 most influential people in the world in 2009. He has also been featured in various media outlets, including The New York Times, The Wall Street Journal, and NPR, for his work on predicting and preventing the spread of infectious diseases.",
    ]
    title = ["Nathan Wolfe", "Nathan Wolfe"]

    result = scorer.factuality_count_reward_func(
        prompts=prompts, completions=completions, title=title
    )
    print(result)
