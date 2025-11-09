# svsync/spatial.py

def get_position_description(box, image_size):
    x0, y0, x1, y1 = map(int, box)
    img_w, img_h = image_size
    xc = (x0 + x1) / 2
    yc = (y0 + y1) / 2

    if xc < img_w / 3:
        horizontal = "on the left"
    elif xc < 2 * img_w / 3:
        horizontal = "in the center"
    else:
        horizontal = "on the right"

    if yc < img_h / 3:
        vertical = "at the top"
    elif yc < 2 * img_h / 3:
        vertical = "in the middle"
    else:
        vertical = "at the bottom"

    return f"{vertical} {horizontal}"

def extract_caption_and_position(hlbb_list, image_size):
    results = []
    for obj in hlbb_list:
        box = obj["box"]
        caption = obj.get("caption", "aucune description")
        position = get_position_description(box, image_size)
        results.append({"position": position, "caption": caption})
    return results
