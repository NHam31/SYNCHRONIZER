# svsync/pipeline.py
import torch
from PIL import Image
import numpy as np
import supervision as sv

from .vision_utils import load_rgb, to_bgr, apply_mask_white_bg, boxes_to_crops_pil
from .spatial import get_position_description
from .hlbb import extract_hl_features

@torch.inference_mode()
def describe_object(image_rgb_np, mask_bool, processor_blip, model_blip, device):
    masked = apply_mask_white_bg(image_rgb_np, mask_bool)
    pil_img = Image.fromarray(masked)
    inputs = processor_blip(pil_img, return_tensors="pt").to(device)
    out = model_blip.generate(**inputs)
    caption = processor_blip.decode(out[0], skip_special_tokens=True)
    return caption

@torch.inference_mode()
def generate_caption_blip(image_pil, processor_blip, model_blip):
    inputs = processor_blip(image_pil, return_tensors="pt").to(model_blip.device)
    output = model_blip.generate(**inputs)
    caption = processor_blip.decode(output[0], skip_special_tokens=True)
    return caption

@torch.inference_mode()
def extract_objects_with_mistral(description: str, gen):
    prompt = (
        "You are an AI assistant that extracts visual objects from image descriptions.\n"
        "From the following text, list only the main visible objects (no colors, no adjectives, no duplicates).\n"
        "Output a comma-separated list in lowercase. End the list with a dot.\n\n"
        f"Description: {description}\n\nObjects:"
    )
    out = gen(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
    object_line = out.split("Objects:")[-1].split(".")[0].strip()
    object_list = [o.strip().lower() for o in object_line.split(",") if o.strip()]
    return list(dict.fromkeys(object_list))  # dédoublonne en gardant l'ordre

import torch as _t

# svsync/pipeline.py
import torch as _t

def _gdino_postprocess_force_thresholds(
    processor, outputs, inputs, image_pil,
    box_threshold: float, text_threshold: float
):
    """
    Post-process universel: récupère les boîtes via processor, PUIS filtre
    manuellement avec box_threshold + text_threshold (quelque soit la version HF).
    Retourne (results:list[dict], counts:dict).
    """
    H, W = image_pil.size[1], image_pil.size[0]

    # 1) On récupère les boîtes sans seuils (ou avec — on s’en fiche, on re-filtre)
    base = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(H, W)]
    )
    r0 = base[0] if isinstance(base, (list, tuple)) else base

    boxes       = r0.get("boxes", None)
    scores      = r0.get("scores", None)
    logits      = r0.get("logits", None)         # parfois présent
    text_labels = r0.get("text_labels", None) or r0.get("labels", None)

    n_raw = int(boxes.shape[0]) if hasattr(boxes, "shape") else (len(boxes) if isinstance(boxes, list) else 0)

    def to_tensor(x):
        if x is None: return None
        return x if _t.is_tensor(x) else _t.tensor(x)

    boxes  = to_tensor(boxes)
    scores = to_tensor(scores)
    logits = to_tensor(logits)

    # 2) On fabrique des scores si absents
    if scores is None and logits is not None:
        scores = _t.sigmoid(logits)
        if scores.ndim > 1:
            scores = scores.max(dim=-1).values

    # 3) Masques de seuils
    if scores is None:
        keep = _t.ones((boxes.size(0),), dtype=_t.bool, device=boxes.device)
    else:
        keep = scores >= float(box_threshold)

    # text_threshold: si logits 2D, on prend le meilleur token
    if logits is not None:
        tscore = _t.sigmoid(logits)
        if tscore.ndim > 1:
            tscore = tscore.max(dim=-1).values
        keep = keep & (tscore >= float(text_threshold))

    # 4) Appliquer le masque
    def apply_keep(x):
        if x is None: return None
        if _t.is_tensor(x):
            return x[keep]
        if isinstance(x, list):
            idx = _t.nonzero(keep, as_tuple=False).squeeze(1).tolist()
            return [x[i] for i in idx]
        return x

    boxes_f       = apply_keep(boxes)
    scores_f      = apply_keep(scores)
    text_labels_f = apply_keep(text_labels)

    n_kept = int(boxes_f.shape[0]) if hasattr(boxes_f, "shape") else (len(boxes_f) if isinstance(boxes_f, list) else 0)

    results = [{
        "boxes":  boxes_f,
        "scores": scores_f,
        "text_labels": text_labels_f
    }]
    counts = {"n_raw": n_raw, "n_kept": n_kept}
    return results, counts



@torch.inference_mode()
def run_pipeline_bytes(
    image_bytes,
    # SAM
    mask_generator,
    # BLIP (caption)
    processor_blip, model_blip,
    # Grounding DINO (HF)
    gdino_processor, gdino_model,
    # Mistral (extraction objets)
     tokenizer=None, mistral_model=None,   # ancienne voie (à ignorer si mistral_gen n'est pas None)
    mistral_gen=None, 
    # seuils
    box_threshold=0.4, text_threshold=0.3, max_text_len=256,
):
    """Pipeline tel que dans ton notebook — sans changer la logique."""
    # 1) lecture image
    image_rgb = load_rgb(image_bytes)               # np RGB
    image_bgr = to_bgr(image_rgb)                   # pour supervision
    image_pil = Image.fromarray(image_rgb)

    # 2) SAM
    sam_result = mask_generator.generate(image_rgb)

    # 3) affichage segmentation (supervision)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # 4) légendes BLIP pour chaque masque (fond blanc)
    labels = []
    for res in sam_result:
        mask = res["segmentation"]
        caption = describe_object(image_rgb, mask, processor_blip, model_blip, model_blip.device)
        labels.append(caption)

    # affectation directe des labels aux detections (comme le notebook)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    detections.labels = labels

    # 5) phrase complète (automatique)
    if len(labels) > 1:
        phrase_complete = "; ".join(labels[:-1]) + " et " + labels[-1]
    elif labels:
        phrase_complete = labels[0]
    else:
        phrase_complete = ""

    # 6) extraction objets (Mistral) à partir de phrase_complete
    if mistral_gen is not None:
        objects = extract_objects_with_mistral(phrase_complete, mistral_gen)
    else:
        # ancienne implémentation si tu l’avais déjà :
        # objects = extract_objects_with_mistral(phrase_complete, tokenizer, mistral_model)
        # si pas disponible, fallback simple :
        if tokenizer is not None and mistral_model is not None:
            objects = extract_objects_with_mistral(phrase_complete, tokenizer, mistral_model)
        else:
            objects = ["object"] if phrase_complete.strip() == "" else list({w for w in phrase_complete.lower().split()})
    text_for_gdino = objects if objects else ["object"]

    # 7) Grounding DINO (HF) — EXACTEMENT comme dans ton snippet
    inputs = gdino_processor(
        images=image_pil,
        text=text_for_gdino,        # liste [...]
        return_tensors="pt",
        truncation=True,
        max_length=max_text_len
    ).to(gdino_model.device)

    outputs = gdino_model(**inputs)
 
    # AVANT: results = gdino_processor.post_process_grounded_object_detection(...)

    results, gdino_counts = _gdino_postprocess_force_thresholds(
        processor=gdino_processor,
        outputs=outputs,
        inputs=inputs,
        image_pil=image_pil,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
    )


    # juste après 'results = ...'
    n_boxes = int(results[0]["boxes"].shape[0]) if results and results[0].get("boxes", None) is not None else 0
    print(f"[GDINO] boxes après seuils: {n_boxes}")



    # 8) HLBB : features + caption par box
    hlbb_list = []
    image_np = image_rgb
    for r in results:
        raw_boxes = r["boxes"].cpu().numpy()
        raw_labels = r.get("text_labels", None)  # si dispo

        for idx, box in enumerate(raw_boxes):
            features = extract_hl_features(image_np, box, image_pil.size)
            crop_pil = image_pil.crop(tuple(map(int, box)))

            caption = generate_caption_blip(crop_pil, processor_blip, model_blip)
            hlbb_list.append({
                "box": [float(v) for v in box.tolist()] if hasattr(box, "tolist") else list(map(float, box)),
                "label": (raw_labels[idx] if raw_labels is not None and idx < len(raw_labels) else None),
                "caption": caption,
                "features": {
                    "color_histogram": [float(h) for h in features["color_histogram"]],
                    "texture_lbp": [float(t) for t in features["texture_lbp"]],
                    "aspect_ratio": float(features["aspect_ratio"]),
                    "relative_area": float(features["relative_area"]),
                }
            })

    return {
        "image_rgb": image_rgb,
        "annotated_image_bgr": annotated_image,
        "sam_count": len(sam_result),
        "labels_by_mask": labels,
        "phrase_complete": phrase_complete,
        "objects_list": objects,          # >>> liste pour GDINO
        "results_gdino": results,
        "hlbb_list": hlbb_list,
        "gdino_counts": gdino_counts
    }
