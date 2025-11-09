# app.py
import io
import json
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import re
import io
import pandas as pd
import streamlit as st

# --- modules "glue" (ton code organisé) ---
from svsync.spatial import extract_caption_and_position
from svsync.pipeline import run_pipeline_bytes
from svsync.display import show_full_image_with_captions, show_crop_with_caption

# --- loaders.py (on le laisse tel quel). On attend ces fonctions : ---
#   - load_sam(ckpt_path) -> mask_generator
#   - load_blip() -> (processor_blip, model_blip)
#   - load_gdino_hf(model_id) -> (gdino_processor, gdino_model)
#   - load_mistral() -> (tokenizer, mistral_model)
try:
    from svsync.loaders import load_sam, load_blip, load_gdino_hf
    from svsync.loaders import load_mistral_pipeline

except Exception as e:
    raise RuntimeError(
        "Chargement de loaders.py échoué. Assure-toi qu'il expose "
        "load_sam, load_blip, load_gdino_hf et load_mistral."
    ) from e


# ----------------- CONFIG UI -----------------
st.set_page_config(page_title="Segma Vision Synchronizer — HLBB", layout="wide")
st.title("Segma Vision Synchronizer — HLBB")

with st.sidebar:
    st.header("⚙️ Paramètres (inférence)")
    gdino_model_id = st.selectbox(
        "Grounding DINO (HF)",
        ["IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny"],
        index=0
    )
    box_thresh = st.slider("Seuil box (score)", 0.0, 0.9, 0.40, 0.01)
    text_thresh = st.slider("Seuil texte", 0.0, 0.9, 0.30, 0.01)
    max_text_len = st.slider("Longueur max du prompt", 32, 512, 256, 32)

    st.divider()
    st.header("🧠 Poids / Checkpoints")
    sam_ckpt_path = st.text_input("SAM checkpoint", "weights/sam_vit_h_4b8939.pth")
    st.caption("Grounding DINO via Transformers – pas de chemin local requis.")

# ----------------- CHARGEMENT DES MODÈLES (cache) -----------------
@st.cache_resource(show_spinner=True)
def _load_models(sam_ckpt: str, gdino_model_id_: str):
    # SAM
    mask_generator = load_sam(sam_ckpt)

    # BLIP (caption)
    processor_blip, model_blip = load_blip()

    # Grounding DINO (HF)
    gdino_processor, gdino_model = load_gdino_hf(gdino_model_id_)

    # >>> Mistral via pipeline (pas de tokenizer/model séparés)
    mistral_gen = load_mistral_pipeline()   # peut aussi prendre hf_token="..."

    return mask_generator, processor_blip, model_blip, gdino_processor, gdino_model, mistral_gen

mask_generator, processor_blip, model_blip, gdino_processor, gdino_model, mistral_gen = _load_models(
    sam_ckpt_path, gdino_model_id
)

# ----------------- UI PRINCIPALE -----------------
st.markdown("### 📷 Image d’entrée")
file = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 1])

if file is not None:
    image_bytes = file.read()

    with st.spinner("Pipeline (SAM → BLIP → Mistral → Grounding DINO → HLBB)…"):
        out = run_pipeline_bytes(
            image_bytes=image_bytes,
            mask_generator=mask_generator,
            processor_blip=processor_blip,
            model_blip=model_blip,
            gdino_processor=gdino_processor,
            gdino_model=gdino_model,
            tokenizer=None,
            mistral_gen=mistral_gen, 
            mistral_model=None,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            max_text_len=max_text_len,
        )

    # ----- Affichage segmentation (SAM) -----
    with col1:
        st.subheader("🧩 Segmentation SAM")
        bgr = out["annotated_image_bgr"]
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Masques SAM : {out['sam_count']}", use_column_width=True)

    # ----- Récap texte -----
    with col2:
        st.subheader("📝 Récap")
        st.markdown(f"- **Masques générés** : {out['sam_count']}")
        st.markdown(f"- **Liste d’objets pour Grounding DINO** : `{out['objects_list']}`")
        st.markdown(f"- **Boîtes GDINO (avant seuils)** : {out['gdino_counts']['n_raw']}")
        st.markdown(f"- **Boîtes GDINO (après seuils)** : {out['gdino_counts']['n_kept']}")

    # ----- Visualisation HLBB complète (matplotlib) -----
    st.subheader("🔎 HLBB — Visualisation globale")
    try:
        img_pil = Image.fromarray(out["image_rgb"])
        fig = show_full_image_with_captions(img_pil, out["hlbb_list"])
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Affichage matplotlib : {e}")
    
    st.subheader("🧭 Positions & captions (avant export)")

    # 1) On récupère positions + captions depuis ta fonction existante
    caption_position_list = extract_caption_and_position(out["hlbb_list"], img_pil.size)

    # 2) Petit nettoyeur pour les libellés de position (évite 'in in the middle', etc.)
    def _tidy_position(pos: str) -> str:
        s = pos.strip().lower()

        # dédoublonner les 'in ' : "in in the middle" -> "in the middle"
        s = re.sub(r"\bin\s+in\b", "in", s)

        # éviter "in on the left/right" -> "on the left/right"
        s = re.sub(r"\bin\s+(on\s+(the\s+)?)", r"\1", s)

        # éviter "in at the top/bottom" -> "at the top/bottom"
        s = re.sub(r"\bin\s+(at\s+(the\s+)?)", r"\1", s)

        # espaces multiples
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    # 3) Fabrique les lignes formatées comme demandé
    lines = []
    for i, item in enumerate(caption_position_list, 1):
        pos = _tidy_position(item.get("position", ""))
        cap = item.get("caption", "aucune description")
        lines.append(f'Objet {i} is {cap}, {pos}.')

    # 4) Affichage dans Streamlit
    st.code("\n".join(lines) if lines else "[ ]", language="text")


    # (Optionnel) Table récap pour vérification rapide
    df_pc = pd.DataFrame(
        [{"index": i+1, "position": _tidy_position(x.get("position","")), "caption": x.get("caption","")} 
        for i, x in enumerate(caption_position_list)]
    )
    st.dataframe(df_pc, use_container_width=True)

    # ---------------- Description finale (paragraphe unique) ----------------
   

    st.subheader("📝 Final description")

    # 1) petits utilitaires (pas de nouvelles dépendances)
    _ARTICLE_RE = re.compile(r"^(a|an|the)\s+", flags=re.I)
    _SPACE_RE   = re.compile(r"\s{2,}")

    def norm_caption(c: str) -> str:
        """Nettoie la caption pour le dédoublonnage et l'affichage."""
        s = (c or "").strip()
        s = _ARTICLE_RE.sub("", s)      # enlève a/an/the au début
        s = s.rstrip(".")               # retire un point final isolé
        s = _SPACE_RE.sub(" ", s)
        return s

    def split_position(pos: str):
        """Extrait (vertical, horizontal) depuis 'at the top in the center', etc."""
        s = (pos or "").lower()
        vert = "in the middle"
        horiz = "in the center"
        if "at the top" in s:       vert = "at the top"
        elif "at the bottom" in s:  vert = "at the bottom"
        if "on the left" in s:      horiz = "on the left"
        elif "on the right" in s:   horiz = "on the right"
        return vert, horiz

    # 2) on regroupe par zone (haut/milieu/bas × gauche/centre/droite)
    BUCKETS = [
        (("at the top",    "on the left"),   "At the top left"),
        (("at the top",    "in the center"), "At the top center"),
        (("at the top",    "on the right"),  "At the top right"),
        (("in the middle", "on the left"),   "In the middle left"),
        (("in the middle", "in the center"), "In the center"),
        (("in the middle", "on the right"),  "In the middle right"),
        (("at the bottom", "on the left"),   "At the bottom left"),
        (("at the bottom", "in the center"), "At the bottom center"),
        (("at the bottom", "on the right"),  "At the bottom right"),
    ]

    bucket_map = {k: [] for k, _ in BUCKETS}
    seen = set()  # pour éliminer les doublons (caption, zone)

    for item in caption_position_list:
        cap = norm_caption(item.get("caption", ""))
        pos = item.get("position", "")
        vert, horiz = split_position(pos)
        key_zone = (vert, horiz)
        if key_zone not in bucket_map or not cap:
            continue
        # dédoublonnage simple par (caption normalisée, zone)
        sig = (cap.lower(), key_zone)
        if sig in seen:
            continue
        seen.add(sig)
        bucket_map[key_zone].append(cap)

    # 3) fabrique des segments lisibles par zone (énumération FR)
    segments = []
    for key_zone, lead in BUCKETS:
        caps = bucket_map[key_zone]
        if not caps:
            continue
        if len(caps) == 1:
            phrase = f"{lead}, {caps[0]}."
        elif len(caps) == 2:
            phrase = f"{lead}, {caps[0]} and {caps[1]}."
        else:
            phrase = f"{lead}, " + ", ".join(caps[:-1]) + f" and {caps[-1]}."
        segments.append(phrase)

    # 4) paragraphe final
    if segments:
        final_paragraph ="The image shows the following elements: " + " ".join(segments)
    else:
        final_paragraph = "No reliable description could be generated."
    st.write(final_paragraph)

    
    # 

    # ----- Export JSON -----
    st.subheader("💾 Export HLBB.json")
    export = {
        "phrase_complete": out["phrase_complete"],
        "objects": out["objects_list"],
        "hlbb": out["hlbb_list"],
    }
    buf = io.BytesIO(json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button(
        label="Télécharger HLBB.json",
        data=buf,
        file_name="HLBB.json",
        mime="application/json",
    )

else:
    st.info("Charge une image (.png / .jpg / .jpeg) pour lancer le pipeline.")
