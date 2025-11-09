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


