# Segma Vision Pro Synchronizer

**Segma Vision Pro Synchronizer** is a multimodal AI pipeline for advanced object detection and interpretation in images. It combines state-of-the-art models like **SAM**, **BLIP**, **Mistral**, and **Grounding DINO** to produce enriched high-level representations of detected objects, called **HLBBs** (High-Level Bounding Boxes).

## Pipeline Overview

This project follows a five-stage intelligent vision workflow:

### 1. Image Segmentation with SAM (Segment Anything Model)
- Segments all objects in an image.
- Outputs individual object masks.

### 2. Captioning Segmented Objects with BLIP
- Each segmented object is passed to [BLIP](w) (Bootstrapped Language Image Pretraining) to generate a caption.
- Captions describe the content and context of each object.

### 3. Keyword Extraction using Mistral
- Captions are processed by the [Mistral](w) language model.
- Extracts **clean, representative keywords** for all detected objects.

### 4. Object Localization with Grounding DINO
- Keywords are used as **text prompts** for [Grounding DINO](w), a zero-shot object detector.
- Returns bounding boxes (BBs) and their confidence scores.

### 5. HLBB: High-Level Bounding Boxes (Multimodal Enrichment)
Each detected object is enriched with:
- **Color features** (RGB histogram)
- **Texture** (Local Binary Pattern)
- **Geometrical descriptors** (aspect ratio, relative area)
- **Object name** (from keywords)
- **Natural language description** of the object

All outputs are stored in a JSON file (`hlbb_output.json`) for further use or integration.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NHam31/SYNCHRONIZER/blob/main/notebooks/sam_Blip_HLBB_clean.ipynb)
##  Project Structure

```bash
SegmaVisionPro/
├── segment/                  # SAM segmentation module
├── captioning/               # BLIP-based captioning per object
├── keyword_extraction/       # Mistral-based NLP keyword extractor
├── detection/                # Grounding DINO inference
├── hlbb/                     # HLBB construction & JSON export
├── utils/                    # Utility functions (filters, image tools, etc.)
├── pic/                      # Sample images
├── hlbb_output.json          # Final enriched object data
└── README.md                 # 📄 This file

```

##  Example Result
   add images
##  Requirements

Install the necessary libraries via:

```bash

pip install torch torchvision transformers opencv-python numpy matplotlib

```
You’ll also need:

- A GPU (recommended)
- Access to Hugging Face models: ```bash facebook/sam```, ```bash Salesforce/blip```, ```bash mistralai/Mistral-7B-Instruct-v0.3 ```, ```bash IDEA-Research/grounding-dino-tiny```

## 🚀 Deployment

This project ships with a ready-to-run **Streamlit** app (English UI).

### Local (recommended)
```bash
# 1) create & activate a virtual env (optional)
python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2) install dependencies
pip install -r requirements.txt

# 3) (optional) download model weights
bash scripts/download_weights.sh

# 4) run the app
# If the app lives at repo root:
streamlit run app.py
# If the app is under ./app:
# cd app && streamlit run app.py
```
### GPU on Windows (WSL2 + NVIDIA)

-Ensure NVIDIA driver is installed on Windows and nvidia-smi works.

-In WSL2, verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
expected output : TRUE
### Streamlit Community Cloud

1. Go to **https://share.streamlit.io** → **Deploy an app**.

2. Select this repo and set:
   - **Main file path**: `app.py` _(or `app/app.py` if your app is in a subfolder)_
   - **(Optional) Secrets** → add your Hugging Face token if needed: `HUGGING_FACE_HUB_TOKEN`

3. Click **Deploy**.

> **Note:** Weights are **not** versioned. If your app needs local checkpoints, use:
> - `scripts/download_weights.sh`, or
> - load from Hugging Face at runtime.



