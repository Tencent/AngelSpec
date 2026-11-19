"""Prompt helpers shared by decoder OPD engines."""


def prefix_media(base_ids, prefix, media):
    """Count completed standalone images/videos, rejecting cuts inside media."""
    if not media:
        return None
    image_depth = video_depth = images = videos = 0
    for token in base_ids[:prefix]:
        if token == 120122:
            video_depth += 1
        elif token == 120123:
            video_depth -= 1
            videos += 1
        elif token == 120118:
            image_depth += 1
        elif token == 120119:
            image_depth -= 1
            images += int(video_depth == 0)
    if image_depth or video_depth:
        raise ValueError("OPD anchor cuts an incomplete media block")
    return {
        "images": (media.get("images") or [])[:images] or None,
        "videos": (media.get("videos") or [])[:videos] or None,
    }
