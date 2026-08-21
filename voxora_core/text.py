"""Unicode (Tamil) text rendering on top of OpenCV frames."""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REGULAR_FONT_PATHS = (
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\NirmalaUI.ttf",
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\latha.ttf",
    "tamil_font.ttf",
)

BOLD_FONT_PATHS = (
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\NirmalB.ttf",
    r"C:\Windows\Fonts\NirmalaUIB.ttf",
) + REGULAR_FONT_PATHS


def load_font(font_size, bold=False):
    """Load the first available Tamil-capable font, falling back to PIL's."""
    for path in BOLD_FONT_PATHS if bold else REGULAR_FONT_PATHS:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith(".ttc"):
                return ImageFont.truetype(path, font_size, index=0)
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width, max_lines):
    """Greedily wrap text to a pixel width, ellipsising past ``max_lines``."""
    lines = []
    current_line = ""

    for word in text.split():
        candidate = word if not current_line else current_line + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] += "..."

    return lines


def draw_tamil_text(
    img,
    text,
    position,
    font_size=24,
    color=(255, 200, 0),
    max_width=650,
    max_lines=3,
    bold=False,
):
    """Draw wrapped Tamil/English text on a BGR frame and return the new frame."""
    font = load_font(font_size, bold=bold)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    x, y = position
    line_spacing = font_size + 14

    for line in wrap_text(draw, text, font, max_width, max_lines):
        draw.text((x, y), line, font=font, fill=color)
        y += line_spacing

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
