# svsync/display.py
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .spatial import get_position_description  # on réutilise ta fonction

def show_crop_with_caption(image_pil: Image.Image, box, caption: str):
    x0, y0, x1, y1 = map(int, box)
    crop = image_pil.crop((x0, y0, x1, y1))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(crop)
    ax.axis("off")
    ax.set_title(caption, fontsize=10)
    fig.tight_layout()
    return fig  # ← IMPORTANT

def show_full_image_with_captions(image_pil: Image.Image, hlbb_list: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_pil)
    image_size = image_pil.size

    for idx, obj in enumerate(hlbb_list):
        box = obj["box"]
        caption = obj.get("caption", "aucune description")
        x0, y0, x1, y1 = map(int, box)

        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        pos_desc = get_position_description(box, image_size)
        text = f"{idx+1}. {pos_desc} : {caption}"

        ax.text(
            x0, max(0, y0 - 10), text,
            fontsize=8, color="white", backgroundcolor="black"
        )

    ax.axis("off")
    fig.tight_layout()
    return fig  # ← IMPORTANT
