# SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents

This repository contains the code and experimental setup for our paper on detecting prompt injection attacks in screenshot-based web agents.

## 🗂️ Repository Structure

```
.
├── detector_image/         # Image-based detectors
│   ├── snapguard.py        # Our proposed SnapGuard detector
│   ├── llava.py            # LLaVA-based detector
│   ├── jailguard.py        # JailGuard baseline
│   ├── ensemble.py         # Ensemble detector
│   └── ...
├── detector_text/           # Text-based baseline detectors
│   ├── promptguard.py      # PromptGuard
│   ├── promptarmor.py      # PromptArmor
│   ├── datasentinel.py     # DataSentinel
│   ├── kad.py              # KAD
│   └── ...
├── format/                  # Visualization scripts for paper figures
│   ├── figure3_roc.py      # ROC curves (Figure 3)
│   ├── figure4_f1_time.py  # F1 vs. Time analysis (Figure 4)
│   ├── figure5_robust.py   # Robustness evaluation (Figure 5)
│   ├── figure6_gradvar.py  # VSI analysis (Figure 6)
│   ├── table1_image.py     # Image detector results (Table 1)
│   └── ...
├── fig/                     # Generated paper figures
├── data/                    # Dataset directory
│   └── image/              # Screenshot images
│       ├── benign/         # Benign samples
│       └── malicious/      # Malicious samples
├── result/                  # Experimental results
├── main_image_split.py      # Main experiment runner
└── README.md
```

## 🔧 Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Tesseract OCR

### Installation

1. Create conda environment from the provided configuration:
```bash
conda env create -f environment.yml
conda activate pidefense
```

2. Install Tesseract OCR (required for SnapGuard detector):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

3. Configure model paths in `detector_image/snapguard.py`:
```python
LLAVA_MODEL_PATH = "YOUR_LLaVA_MODEL_PATH"  # For LLaVA-based detectors
LLAMA3_MODEL_PATH = "YOUR_LLaMA3_MODEL_PATH"  # For text-based detectors
```

### Dataset

This work uses the **WAInjectBench** benchmark for evaluation:
- Repository: https://github.com/Norrrrrrr-lyn/WAInjectBench/
- Place the dataset in `data/image/` directory with the following structure:
  ```
  data/image/
  ├── benign/
  │   ├── dataset1/
  │   ├── dataset2/
  │   └── ...
  └── malicious/
      ├── dataset1/
      ├── dataset2/
      └── ...
  ```

## 🚀 Quick Start

### Running SnapGuard Detector

```bash
python main_image_split.py \
    --data_dir data/image \
    --detector snapguard \
    --result_dir result/image_split \
    --gpu 0
```

### Running Baseline Detectors

```bash
# LLaVA-based detector
python main_image_split.py --detector llava-1.5-7b-prompt --gpu 0

# JailGuard
python main_image_split.py --detector jailguard --gpu 0

# Ensemble detector
python main_image_split.py --detector ensemble --gpu 0
```

### Parameters

- `--data_dir`: Path to the dataset directory (default: `data/image`)
- `--detector`: Detector name (see [Available Detectors](#available-detectors))
- `--result_dir`: Output directory for results (default: `result/image_split`)
- `--gpu`: GPU device ID (default: `3`)

## 🔍 Detection Methods

### SnapGuard (Image-Based Detection)

Our proposed method combines two complementary detection strategies:

1. **VSI (Visual Saliency Index)**: Computes gradient variance to identify visually suspicious images
   - Low VSI values indicate potential steganographic or adversarial content
   - Threshold: 4450.0 (configurable in `snapguard.py`)

2. **Attack Action Pattern Detection**: Uses OCR + pattern matching to identify malicious behaviors:
   - **Position-based clicks**: `click[123]`, `click here`, etc.
   - **Credential requests**: API keys, passwords, tokens
   - **Link invitations**: Suspicious URL redirects
   - **Control instructions**: `ignore`, `don't ask`, `without user`, etc.
   - **Password input**: Password entry requests

### Baseline Detectors

**Image-based:**
- `llava-1.5-7b-prompt`: LLaVA with prompt-based detection
- `llava-1.5-7b-ft`: Fine-tuned LLaVA
- `jailguard`: JailGuard detector
- `ensemble`: Ensemble of multiple detectors

**Text-based (for comparison):**
- `promptguard`: PromptGuard
- `promptarmor`: PromptArmor
- `datasentinel`: DataSentinel
- `kad`: KAD (Knowledge-Augmented Detection)


## 📈 Results Format

Results are saved in JSONL format in the `result/` directory:

**Main results** (`{detector_name}.jsonl`):
```json
{
  "data_name": "dataset_name",
  "label": "malicious",
  "tpr": 0.95,
  "detect_ids": [1, 5, 7, ...],
  "total_num": 100,
  "time_sec": 12.34,
  "gpu_mem_mb": 2048
}
```

**Debug information** (`{detector_name}_debug.jsonl`):
```json
{
  "data_name": "dataset_name",
  "label": "malicious",
  "num_triggered": 95,
  "num_low_gradvar": 30,
  "num_attack_action": 65,
  "action_type_counter": {
    "position_click": 25,
    "control_instruction": 20,
    "credential_request": 15,
    ...
  }
}
```

## 🔬 Available Detectors

### Image Detectors

| Detector | Description | File |
|----------|-------------|------|
| `SnapGuard` | Our proposed method (full) | `detector_image/snapguard.py` |
| `llava-1.5-7b-prompt` | LLaVA with detection prompt | `detector_image/llava.py` |
| `llava-1.5-7b-ft` | Fine-tuned LLaVA | `detector_image/llava.py` |
| `jailguard` | JailGuard baseline | `detector_image/jailguard.py` |
| `ensemble` | Ensemble detector | `detector_image/ensemble.py` |
| `embedding-i` | Embedding-based detector | `detector_image/embedding-i.py` |
| `gpt-4o-prompt` | GPT-4o with detection prompt | `detector_image/gpt-4o-prompt.py` |

### Text Detectors (Baselines)

| Detector | Description | File |
|----------|-------------|------|
| `promptguard` | PromptGuard | `detector_text/promptguard.py` |
| `promptarmor` | PromptArmor | `detector_text/promptarmor.py` |
| `datasentinel` | DataSentinel | `detector_text/datasentinel.py` |
| `kad` | Knowledge-Augmented Detection | `detector_text/kad.py` |
| `embedding-t` | Text embedding-based | `detector_text/embedding-t.py` |
| `ensemble` | Text ensemble | `detector_text/ensemble.py` |


## 🔗 Related Resources

- **WAInjectBench**: https://github.com/Norrrrrrr-lyn/WAInjectBench/
- Benchmark for evaluating prompt injection detection methods on web agents

