from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import os
import re
import torch
import numpy as np
import cv2
import pytesseract
from PIL import Image

# ============================================================
# Environment
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ============================================================
# Config
# ============================================================
LLAVA_MODEL_PATH = "YOUR_LLaVA_MODEL_PATH"
LLAMA3_MODEL_PATH = "YOUR_LLaMA3_MODEL_PATH"

DEBUG_MAX_IMAGES = None   # e.g. 20 / 50 / None
VSI_THRESHOLD = 4450.0

# ============================================================
# OCR normalization
# ============================================================
def normalize_for_action(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\[\]\(\)]", "", s)
    return s

# ============================================================
# Attack pattern detectors
# ============================================================
def is_position_click(text: str) -> bool:
    """
    Detect position-based clicks and click inducement
    under aggressive OCR normalization (no spaces).
    """

    patterns = [
        r"click[\[\(]\d+[\]\)]",
        r"clickhere",
        r"pleaseclick",
        r"clickto",
        r"clickthe",
        r"taphere",
        r"presshere",
    ]

    return any(re.search(p, text) for p in patterns)


def is_password_input(text: str) -> bool:
    return (
        any(k in text for k in ["password", "passwd", "pwd"])
        and any(v in text for v in ["enter", "input", "type"])
    )

def is_link_invitation(text: str) -> bool:
    return (
        "http" in text
        or "www" in text
        or ("link" in text and any(v in text for v in ["click", "open", "visit"]))
    )

def is_control_instruction(text: str) -> bool:
    """
    Detect control / override instructions under
    aggressive OCR normalization (no spaces).
    """

    keywords = [
        "alert",
        "ignore",
        "follow",
        "donotask",
        "dontask",
        "withoutasking",
        "withoutuser",
        "donotinform",
        "noneedtoask",
        "dontinform",
        "noaskuser",
        "askagainmanytimes",
    ]

    return any(k in text for k in keywords)


def is_api_key_request(norm_line: str) -> bool:
    """
    Detect requests for API keys / credentials under OCR noise.
    """

    credential_patterns = [
        r"api\s*key",
        r"apikey",
        r"access\s*token",
        r"auth\s*token",
        r"secret",
        r"private\s*key",
    ]

    request_patterns = [
        r"share",
        r"provide",
        r"send",
        r"give",
        r"pass",
        r"supply",
        r"need",
        r"require",
    ]

    has_credential = any(re.search(p, norm_line) for p in credential_patterns)
    has_request = any(re.search(p, norm_line) for p in request_patterns)

    if has_credential and has_request:
        return True

    if re.search(r"\bsk-[a-z0-9]{10,}\b", norm_line):
        return True

    return False



def detect_malicious_actions(ocr_text: str) -> List[Dict[str, str]]:
    """
    Return detailed malicious action matches.
    Each item:
      {
        "action_type": str,
        "matched_text": str
      }
    """
    hits = []

    for line in ocr_text.splitlines():
        norm = normalize_for_action(line)

        if is_position_click(norm):
            hits.append({
                "action_type": "position_click",
                "matched_text": line
            })

        if is_password_input(norm):
            hits.append({
                "action_type": "password_input",
                "matched_text": line
            })

        if is_link_invitation(norm):
            hits.append({
                "action_type": "link_invitation",
                "matched_text": line
            })

        if is_control_instruction(norm):
            hits.append({
                "action_type": "control_instruction",
                "matched_text": line
            })

        if is_api_key_request(norm):
            hits.append({
                "action_type": "credential_request",
                "matched_text": line
            })

    return hits


# ============================================================
# VSI (unchanged)
# ============================================================
def compute_vsi(pil_image: Image.Image) -> float:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag.var()

# ============================================================
# OCR utilities (unchanged)
# ============================================================
def simple_ocr(pil_image: Image.Image) -> List[str]:
    gray = np.array(pil_image.convert("L"))
    txt = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return [l.strip() for l in txt.splitlines() if l.strip()]

def invert_near_white(pil_image: Image.Image, thresh: int = 240) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > thresh
    inv = img.copy()
    inv[mask] = 255 - inv[mask]
    return Image.fromarray(inv)

def dual_pass_ocr(pil_image: Image.Image) -> str:
    lines = simple_ocr(pil_image) + simple_ocr(invert_near_white(pil_image))
    uniq, seen = [], set()
    for l in lines:
        k = l.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(l)
    return "\n".join(uniq)



def detect(folder_path: str):
    folder = Path(folder_path)
    detect_ids = []
    debug_info = {}

    img_files = sorted(
        p for p in folder.glob("*")
        if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]
    )

    if DEBUG_MAX_IMAGES:
        img_files = img_files[:DEBUG_MAX_IMAGES]

    for img_file in tqdm(img_files):
        image = Image.open(img_file).convert("RGB")
        image_id = img_file.stem

        record = {}

        # =========================
        # 1. vsi (primary gate)
        # =========================
        gv = compute_vsi(image)
        record["vsi"] = gv

        if gv < VSI_THRESHOLD:
            # ---- short-circuit trigger ----
            record["trigger_reason"] = "low_vsi"
            record["ocr_skipped"] = True

            detect_ids.append(image_id)
            debug_info[image_id] = record
            continue   # ❗ skip OCR & action detection

        # =========================
        # 2. OCR (only if needed)
        # =========================
        ocr_text = dual_pass_ocr(image)
        record["ocr_text"] = ocr_text
        record["ocr_skipped"] = False

        # =========================
        # 3. Action detection
        # =========================
        action_hits = detect_malicious_actions(ocr_text)
        malicious = len(action_hits) > 0

        record["malicious_action"] = malicious
        record["action_hits"] = action_hits

        if malicious:
            record["trigger_reason"] = "attack_action_pattern"
            record["trigger_action_types"] = sorted(
                set(h["action_type"] for h in action_hits)
            )
            record["trigger_action_details"] = action_hits

            detect_ids.append(image_id)

        debug_info[image_id] = record

    return detect_ids, debug_info

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    pass
