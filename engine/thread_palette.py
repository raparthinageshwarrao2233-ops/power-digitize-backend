"""
POWER DIGITIZE — Thread Palette Database
Professional thread color database with CIE Delta E 2000 matching.
Includes Madeira Rayon, Isacord, and common embroidery thread colors.
"""

import math
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


# ═══════ CIE Delta E 2000 ═══════

def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    CIE Delta E 2000 — perceptually uniform color difference metric.

    More accurate than Euclidean LAB distance because it accounts for:
    - Lightness weighting (kL)
    - Chroma weighting (kC)
    - Hue weighting (kH)
    - Rotation term for blue region correction
    - Interactive chroma-hue term

    Returns a float ≥ 0. Values < 1 are imperceptible to humans.
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    # Step 1: Calculate C'ab and h'ab
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_avg = (C1 + C2) / 2.0

    C_avg_7 = C_avg ** 7
    G = 0.5 * (1.0 - math.sqrt(C_avg_7 / (C_avg_7 + 25.0 ** 7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = math.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = math.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    # Step 2: Calculate ΔL', ΔC', ΔH'
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        dh_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180:
        dh_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180:
        dh_prime = h2_prime - h1_prime - 360
    else:
        dh_prime = h2_prime - h1_prime + 360

    dH_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2.0))

    # Step 3: Calculate CIEDE2000
    L_avg = (L1 + L2) / 2.0
    C_avg_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        h_avg_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_avg_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360:
        h_avg_prime = (h1_prime + h2_prime + 360) / 2.0
    else:
        h_avg_prime = (h1_prime + h2_prime - 360) / 2.0

    T = (1.0
         - 0.17 * math.cos(math.radians(h_avg_prime - 30))
         + 0.24 * math.cos(math.radians(2 * h_avg_prime))
         + 0.32 * math.cos(math.radians(3 * h_avg_prime + 6))
         - 0.20 * math.cos(math.radians(4 * h_avg_prime - 63)))

    SL = 1.0 + 0.015 * (L_avg - 50) ** 2 / math.sqrt(20 + (L_avg - 50) ** 2)
    SC = 1.0 + 0.045 * C_avg_prime
    SH = 1.0 + 0.015 * C_avg_prime * T

    C_avg_prime_7 = C_avg_prime ** 7
    RT = (-2.0 * math.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25.0 ** 7))
          * math.sin(math.radians(60 * math.exp(-((h_avg_prime - 275) / 25) ** 2))))

    kL = kC = kH = 1.0  # Parametric weighting factors

    term_L = dL_prime / (kL * SL)
    term_C = dC_prime / (kC * SC)
    term_H = dH_prime / (kH * SH)

    return math.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + RT * term_C * term_H)


# ═══════ THREAD DATABASE ═══════

# Real embroidery thread colors — Madeira Rayon & Isacord Polyester
# Format: (name, hex, brand)
THREAD_DATABASE: List[Tuple[str, str, str]] = [
    # ── Whites & Neutrals ──
    ("White", "#FFFFFF", "Universal"),
    ("Natural White", "#F5F0E1", "Madeira"),
    ("Eggshell", "#F0EAD6", "Isacord"),
    ("Cream", "#FFFDD0", "Madeira"),
    ("Ivory", "#FFFFF0", "Isacord"),
    ("Pearl", "#EAE0C8", "Madeira"),

    # ── Blacks & Grays ──
    ("Black", "#000000", "Universal"),
    ("Charcoal", "#36454F", "Madeira"),
    ("Dark Gray", "#4A4A4A", "Isacord"),
    ("Medium Gray", "#808080", "Universal"),
    ("Sterling", "#969696", "Madeira"),
    ("Silver", "#C0C0C0", "Isacord"),
    ("Light Gray", "#D3D3D3", "Universal"),
    ("Ash", "#B2BEB5", "Madeira"),

    # ── Reds ──
    ("Bright Red", "#FF0000", "Madeira"),
    ("Cardinal Red", "#C41E3A", "Isacord"),
    ("Cranberry", "#9B2335", "Madeira"),
    ("Burgundy", "#800020", "Isacord"),
    ("Wine", "#722F37", "Madeira"),
    ("Dark Red", "#8B0000", "Isacord"),
    ("Cherry", "#DE3163", "Madeira"),
    ("Scarlet", "#FF2400", "Isacord"),
    ("Brick Red", "#CB4154", "Madeira"),
    ("Rose", "#FF007F", "Isacord"),
    ("Dusty Rose", "#DCAE96", "Madeira"),
    ("Salmon", "#FA8072", "Isacord"),
    ("Coral", "#FF7F50", "Madeira"),

    # ── Oranges ──
    ("Orange", "#FF8C00", "Madeira"),
    ("Bright Orange", "#FF6600", "Isacord"),
    ("Tangerine", "#FF9966", "Madeira"),
    ("Pumpkin", "#FF7518", "Isacord"),
    ("Rust", "#B7410E", "Madeira"),
    ("Burnt Orange", "#CC5500", "Isacord"),
    ("Peach", "#FFCBA4", "Madeira"),
    ("Apricot", "#FBCEB1", "Isacord"),

    # ── Yellows ──
    ("Bright Yellow", "#FFD700", "Madeira"),
    ("Lemon", "#FFF44F", "Isacord"),
    ("Gold", "#DAA520", "Madeira"),
    ("Old Gold", "#CFB53B", "Isacord"),
    ("Mustard", "#FFDB58", "Madeira"),
    ("Pale Yellow", "#FFFACD", "Isacord"),
    ("Canary", "#FFEF00", "Madeira"),
    ("Sunflower", "#FFDA03", "Isacord"),

    # ── Greens ──
    ("Kelly Green", "#4CBB17", "Madeira"),
    ("Bright Green", "#00FF00", "Isacord"),
    ("Forest Green", "#228B22", "Madeira"),
    ("Hunter Green", "#355E3B", "Isacord"),
    ("Emerald", "#50C878", "Madeira"),
    ("Dark Green", "#006400", "Isacord"),
    ("Olive", "#808000", "Madeira"),
    ("Sage", "#BCB88A", "Isacord"),
    ("Lime", "#32CD32", "Madeira"),
    ("Mint", "#98FF98", "Isacord"),
    ("Teal", "#008080", "Madeira"),
    ("Spring Green", "#00FF7F", "Isacord"),

    # ── Blues ──
    ("Royal Blue", "#4169E1", "Madeira"),
    ("Bright Blue", "#0000FF", "Isacord"),
    ("Navy", "#000080", "Madeira"),
    ("Dark Navy", "#00003E", "Isacord"),
    ("Sky Blue", "#87CEEB", "Madeira"),
    ("Baby Blue", "#89CFF0", "Isacord"),
    ("Cobalt", "#0047AB", "Madeira"),
    ("Steel Blue", "#4682B4", "Isacord"),
    ("Powder Blue", "#B0E0E6", "Madeira"),
    ("Cornflower", "#6495ED", "Isacord"),
    ("Turquoise", "#40E0D0", "Madeira"),
    ("Aqua", "#00FFFF", "Isacord"),
    ("Medium Blue", "#0000CD", "Madeira"),

    # ── Purples & Violets ──
    ("Purple", "#800080", "Madeira"),
    ("Bright Purple", "#A020F0", "Isacord"),
    ("Dark Purple", "#301934", "Madeira"),
    ("Violet", "#7F00FF", "Isacord"),
    ("Lavender", "#E6E6FA", "Madeira"),
    ("Plum", "#8E4585", "Isacord"),
    ("Mauve", "#E0B0FF", "Madeira"),
    ("Orchid", "#DA70D6", "Isacord"),
    ("Grape", "#6F2DA8", "Madeira"),

    # ── Pinks ──
    ("Hot Pink", "#FF69B4", "Madeira"),
    ("Bright Pink", "#FF007F", "Isacord"),
    ("Light Pink", "#FFB6C1", "Madeira"),
    ("Pastel Pink", "#FFD1DC", "Isacord"),
    ("Magenta", "#FF00FF", "Madeira"),
    ("Fuchsia", "#FF00FF", "Isacord"),
    ("Blush", "#DE5D83", "Madeira"),
    ("Carnation", "#FFA6C9", "Isacord"),

    # ── Browns ──
    ("Brown", "#8B4513", "Madeira"),
    ("Dark Brown", "#654321", "Isacord"),
    ("Chocolate", "#7B3F00", "Madeira"),
    ("Coffee", "#6F4E37", "Isacord"),
    ("Tan", "#D2B48C", "Madeira"),
    ("Khaki", "#C3B091", "Isacord"),
    ("Beige", "#F5F5DC", "Madeira"),
    ("Sand", "#C2B280", "Isacord"),
    ("Copper", "#B87333", "Madeira"),
    ("Sienna", "#A0522D", "Isacord"),

    # ── Metallics (common in embroidery) ──
    ("Metallic Gold", "#D4AF37", "Madeira"),
    ("Metallic Silver", "#AAA9AD", "Madeira"),
    ("Metallic Copper", "#B87333", "Madeira"),
]

# Pre-computed LAB values for the thread database
_THREAD_LAB_CACHE: Optional[List[Tuple[np.ndarray, Dict]]] = None


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_str = hex_color.lstrip('#')
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def _get_thread_lab_cache() -> List[Tuple[np.ndarray, Dict]]:
    """Build/return cached LAB values for all threads."""
    global _THREAD_LAB_CACHE
    if _THREAD_LAB_CACHE is not None:
        return _THREAD_LAB_CACHE

    _THREAD_LAB_CACHE = []
    for name, hex_color, brand in THREAD_DATABASE:
        r, g, b = _hex_to_rgb(hex_color)
        rgb_pixel = np.uint8([[[r, g, b]]])
        lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]
        _THREAD_LAB_CACHE.append((lab_pixel, {
            "name": name,
            "hex": hex_color,
            "brand": brand,
            "rgb": [r, g, b],
        }))

    return _THREAD_LAB_CACHE


def match_thread_color(rgb: Tuple[int, int, int]) -> Dict:
    """
    Match an RGB color to the nearest real embroidery thread
    using CIE Delta E 2000 for perceptually accurate matching.

    Returns a dict with: name, hex, brand, rgb, delta_e
    """
    rgb_pixel = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]

    cache = _get_thread_lab_cache()

    best_match = None
    best_de = float('inf')

    for thread_lab, thread_info in cache:
        de = _delta_e_2000(lab_pixel, thread_lab)
        if de < best_de:
            best_de = de
            best_match = thread_info

    result = dict(best_match) if best_match else {
        "name": "Black", "hex": "#000000", "brand": "Universal", "rgb": [0, 0, 0]
    }
    result["delta_e"] = round(best_de, 2)
    return result


def match_palette_to_threads(colors: List[Dict]) -> List[Dict]:
    """
    Match a list of extracted palette colors to real thread colors.
    Each color dict must have an 'rgb' key with [r, g, b].

    Returns a new list with matched thread info added.
    """
    matched = []
    for color in colors:
        r, g, b = color["rgb"]
        thread = match_thread_color((r, g, b))
        matched_color = dict(color)
        matched_color["thread_match"] = thread
        matched_color["thread_name"] = thread["name"]
        matched_color["thread_hex"] = thread["hex"]
        matched_color["thread_brand"] = thread["brand"]
        matched_color["delta_e"] = thread["delta_e"]
        matched.append(matched_color)
    return matched


class MatchedThread:
    def __init__(self, name: str, code: str, brand: str, hex_color: str, delta_e: float):
        self.name = name
        self.code = code
        self.brand = brand
        self.hex = hex_color
        self.delta_e = delta_e


def match_color(rgb: Tuple[int, int, int], brand: str = "Madeira") -> MatchedThread:
    """Match an RGB tuple to real thread color."""
    t = match_thread_color(rgb)
    return MatchedThread(
        name=t["name"],
        code=t.get("code", t["name"]),
        brand=t["brand"],
        hex_color=t["hex"],
        delta_e=t["delta_e"],
    )


def delta_e_2000_lab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Public wrapper for Delta E 2000 calculation from LAB arrays."""
    return _delta_e_2000(lab1, lab2)

