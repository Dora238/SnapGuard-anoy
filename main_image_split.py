import os
import json
import argparse
import time
from pathlib import Path
from typing import List
from collections import Counter

import torch


# ======================================================
# Detector loader
# ======================================================
def load_detector(detector_name: str):
    import importlib

    if detector_name in ["llava-1.5-7b-prompt", "llava-1.5-7b-ft"]:
        return importlib.import_module("detector_image.llava")

    try:
        return importlib.import_module(f"detector_image.{detector_name}")
    except ImportError:
        raise ValueError(
            f"Detector {detector_name} not found. "
            f"Make sure detector_image/{detector_name}.py exists."
        )


# ======================================================
# Helper: normalize detect_ids
# ======================================================
def normalize_detect_ids(detect_ids_raw) -> List[int]:
    """
    Normalize detect_ids into List[int].
    Input must be an iterable of ids (NOT a tuple of (ids, debug)).
    """
    clean_ids = []

    for item in detect_ids_raw:
        if item is None:
            continue

        if isinstance(item, int):
            clean_ids.append(item)

        elif isinstance(item, str):
            clean_ids.append(int(item))

        elif isinstance(item, (list, tuple)):
            if len(item) == 0:
                continue
            clean_ids.append(int(item[0]))

        elif isinstance(item, dict):
            if "id" in item:
                clean_ids.append(int(item["id"]))

        else:
            raise TypeError(f"Unknown detect_id type: {type(item)}")

    return clean_ids


# ======================================================
# Core processing per dataset (sub_folder)
# ======================================================
def process_folder(
    folder_path: Path,
    detector,
    detector_name: str,
    is_malicious: bool,
    result_fp,
    debug_fp
):
    data_name = folder_path.name
    label = "malicious" if is_malicious else "benign"

    # ============ timing & memory ============
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ==================================================
    # Run detector
    # ==================================================
    if detector_name in ["llava-1.5-7b-prompt", "llava-1.5-7b-ft"]:
        ret = detector.detect(
            str(folder_path),
            detector_name=detector_name
        )
    else:
        ret = detector.detect(str(folder_path))

    if isinstance(ret, tuple):
        detect_ids_raw, debug_info = ret
    else:
        detect_ids_raw = ret
        debug_info = None

    elapsed = round(time.time() - start_time, 4)
    peak_mem = (
        round(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if torch.cuda.is_available()
        else 0
    )

    # ============ normalize detect_ids ============
    detect_ids = normalize_detect_ids(detect_ids_raw)
    total_num = len([p for p in folder_path.iterdir() if p.is_file()])

    rate_key = "tpr" if is_malicious else "fpr"
    rate_value = round(len(detect_ids) / total_num, 4) if total_num else 0.0

    # ============ write main result ============
    result_entry = {
        "data_name": data_name,
        "label": label,
        rate_key: rate_value,
        "detect_ids": detect_ids,
        "total_num": total_num,
        "time_sec": elapsed,
        "gpu_mem_mb": peak_mem,
    }
    result_fp.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
    result_fp.flush()

    # ==================================================
    # Debug summary (NEW, aligned with detector)
    # ==================================================
    if debug_info is not None:
        trigger_cnt = 0
        low_gradvar_cnt = 0
        attack_action_cnt = 0
        action_type_counter = Counter()

        for rec in debug_info.values():
            reason = rec.get("trigger_reason")
            if not reason:
                continue

            trigger_cnt += 1

            if reason == "low_gradvar":
                low_gradvar_cnt += 1

            elif reason == "attack_action_pattern":
                attack_action_cnt += 1
                for h in rec.get("action_hits", []):
                    action_type_counter[h["action_type"]] += 1

        debug_entry = {
            "data_name": data_name,
            "label": label,
            "num_triggered": trigger_cnt,
            "num_low_gradvar": low_gradvar_cnt,
            "num_attack_action": attack_action_cnt,
            "action_type_counter": dict(action_type_counter),
        }

        debug_fp.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")
        debug_fp.flush()


# ======================================================
# Experiment runner
# ======================================================
def run_experiment(data_dir: str, detector_name: str, result_dir: str, gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    detector = load_detector(detector_name)

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{detector_name}.jsonl"
    debug_path = result_dir / f"{detector_name}_debug.jsonl"

    with open(result_path, "a", encoding="utf-8") as result_fp, \
         open(debug_path, "a", encoding="utf-8") as debug_fp:

        for folder_name in ["benign", "malicious"]:
        # for folder_name in ["malicious"]:
            parent_path = data_dir / folder_name
            if not parent_path.exists():
                continue

            for sub_folder in sorted(parent_path.iterdir()):
                if not sub_folder.is_dir():
                    continue

                process_folder(
                    folder_path=sub_folder,
                    detector=detector,
                    detector_name=detector_name,
                    is_malicious=(folder_name == "malicious"),
                    result_fp=result_fp,
                    debug_fp=debug_fp
                )


# ======================================================
# Entry
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main Experiment Framework for Image Detectors"
    )

    parser.add_argument("--data_dir", type=str, default="data/image")
    parser.add_argument("--detector", type=str, default="snapguard")
    parser.add_argument("--result_dir", type=str, default="result/image_split")
    parser.add_argument("--gpu", type=str, default="3")

    args = parser.parse_args()

    run_experiment(
        args.data_dir,
        args.detector,
        args.result_dir,
        args.gpu
    )
