import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import os
import sys

# add Open-Prompt-Injection to sys.path
sys.path.append("YOUR_OPEN_PROMPT_INJECTION_PATH")
from OpenPromptInjection.utils import open_config
from OpenPromptInjection import DataSentinelDetector

CONFIG_PATH = "YOUR_OPEN_PROMPT_INJECTION_PATH/configs/model_configs/mistral_config.json"
config = open_config(CONFIG_PATH)
config["params"]["ft_path"] = "YOUR_OPEN_PROMPT_INJECTION_PATH/DataSentinel_Models/detector_large/detector_large/checkpoint-5000"

# Set GPU to 0 to match the current environment
config["params"]["gpus"] = ["0"]
config["params"]["device"] = "cuda:0"

detector_model = DataSentinelDetector(config)


def detect(file_path: str) -> List[int]:
    """
    DataSentinel detector for JSONL text files.

    Args:
        file_path (str): path to a JSONL file, where each line is:
                         {"id": int, "text": str}

    Returns:
        List[int]: list of IDs detected as malicious (detector output == 1).
    """
    detect_ids = []

    # count lines for tqdm
    total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

    with open(file_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total_lines,
                         desc=f"DataSentinel {Path(file_path).name}", ncols=80):
            try:
                item = json.loads(line)
                text_id = item.get("id")
                text = item.get("text", "").strip()

                if text_id is None or not text:
                    continue

                result = detector_model.detect(text)
                if result == 1:
                    detect_ids.append(text_id)

            except Exception as e:
                print(f"[WARNING] Skipped invalid line in {file_path}: {e}")

    return detect_ids
