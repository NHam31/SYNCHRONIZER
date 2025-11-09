import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import (AutoProcessor, BlipForConditionalGeneration,
                          AutoModelForZeroShotObjectDetection, AutoImageProcessor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam(ckpt_path: str):
    sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
    sam.to(DEVICE)
    return SamAutomaticMaskGenerator(sam)

def load_blip():
    proc = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    return proc, model

# --- Grounding DINO (via paquet groundingdino) ---


def load_gdino_hf(model_id: str = "IDEA-Research/grounding-dino-tiny"):
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
    return proc, model


# --- Mistral via HF pipeline (pas de tokenizer à gérer) ---
from transformers import pipeline
from huggingface_hub import login
import os

def load_mistral_pipeline(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", hf_token: str | None = None):
    """
    Retourne une pipeline text-generation pour Mistral.
    Pas de tokenizer/model séparés à manipuler dans ton code.
    """
    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token)
        except Exception:
            pass  # déjà loggé / offline

    gen = pipeline(
        task="text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    return gen  # <- à utiliser comme 'gen(...)'

