"""
POWER DIGITIZE — Monogram Engine
Generates monogram embroidery designs.
"""

import math
from typing import Dict, Any, List
from engine.stitch_generators import StitchBlock, SatinStitchGenerator
from engine.text_engine import STROKE_FONT


class MonogramEngine:
    """Generate monogram embroidery designs."""

    def __init__(self):
        self.satin_gen = SatinStitchGenerator()

    def generate(
        self,
        letters: str,
        design_id: str,
        layout: str = "single",
        size_mm: float = 50.0,
        color: str = "#ffffff",
    ) -> Dict[str, Any]:
        """Generate a monogram design."""

        if layout == "single":
            blocks = self._single_letter(letters[0] if letters else "A", size_mm, color)
        elif layout == "two-letter":
            l = letters[:2] if len(letters) >= 2 else letters + "A"
            blocks = self._two_letter(l, size_mm, color)
        elif layout == "three-letter":
            l = letters[:3] if len(letters) >= 3 else (letters + "AA")[:3]
            blocks = self._three_letter(l, size_mm, color)
        elif layout == "circular":
            blocks = self._circular(letters[0] if letters else "A", size_mm, color)
        elif layout == "badge":
            blocks = self._badge(letters[0] if letters else "A", size_mm, color)
        else:
            blocks = self._single_letter(letters[0] if letters else "A", size_mm, color)

        total_stitches = sum(b.stitch_count for b in blocks)
        thread_length = 0
        for b in blocks:
            for i in range(1, len(b.stitches)):
                s0, s1 = b.stitches[i-1], b.stitches[i]
                thread_length += math.sqrt((s1.x-s0.x)**2 + (s1.y-s0.y)**2)
            transitions = max(0, len(blocks) - 1)

        return {
            "stitch_data": blocks,
            "stats": {
                "stitch_count": total_stitches,
                "jump_count": transitions,
                "trim_count": transitions,
                "color_count": 1,
                "width": round(size_mm, 1),
                "height": round(size_mm, 1),
                "thread_length": round(thread_length / 1000, 1),
                "estimated_time": max(1, round(total_stitches / 800)),
            },
        }

    def _single_letter(self, letter: str, size: float, color: str) -> List[StitchBlock]:
        """Single large letter monogram."""
        scale = size / 14.0
        strokes = STROKE_FONT.get(letter.upper(), STROKE_FONT.get('A', []))
        blocks = []

        offset = -5 * scale  # Center

        for stroke in strokes:
            points = [(p[0] * scale + offset, p[1] * scale + offset) for p in stroke]
            if len(points) >= 2:
                block = self.satin_gen.generate(
                    center_points=points,
                    width=size * 0.12,
                    density=0.3,
                    color=color,
                    color_name="Thread",
                    name=f"Monogram '{letter}'"
                )
                if block.stitch_count > 0:
                    blocks.append(block)

        # Add decorative circle border
        border = self._circle_border(size / 2, color)
        if border.stitch_count > 0:
            blocks.append(border)

        return blocks

    def _two_letter(self, letters: str, size: float, color: str) -> List[StitchBlock]:
        """Two-letter monogram."""
        blocks = []
        scale = size / 14.0 * 0.7
        gap = size * 0.15

        for i, letter in enumerate(letters[:2]):
            strokes = STROKE_FONT.get(letter.upper(), [])
            offset_x = -gap + i * gap * 2 - 5 * scale
            offset_y = -5 * scale

            for stroke in strokes:
                points = [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in stroke]
                if len(points) >= 2:
                    block = self.satin_gen.generate(
                        center_points=points, width=size * 0.08,
                        density=0.3, color=color, color_name="Thread",
                        name=f"Monogram '{letter}'"
                    )
                    if block.stitch_count > 0:
                        blocks.append(block)

        return blocks

    def _three_letter(self, letters: str, size: float, color: str) -> List[StitchBlock]:
        """Three-letter monogram (center letter larger)."""
        blocks = []

        for i, letter in enumerate(letters[:3]):
            is_center = i == 1
            letter_scale = size / 14.0 * (0.8 if is_center else 0.55)
            strokes = STROKE_FONT.get(letter.upper(), [])

            offset_x = (i - 1) * size * 0.35 - 5 * letter_scale
            offset_y = -5 * letter_scale + (0 if is_center else size * 0.1)

            for stroke in strokes:
                points = [(p[0] * letter_scale + offset_x, p[1] * letter_scale + offset_y) for p in stroke]
                if len(points) >= 2:
                    block = self.satin_gen.generate(
                        center_points=points,
                        width=size * (0.1 if is_center else 0.06),
                        density=0.3, color=color, color_name="Thread",
                        name=f"Monogram '{letter}'"
                    )
                    if block.stitch_count > 0:
                        blocks.append(block)

        return blocks

    def _circular(self, letter: str, size: float, color: str) -> List[StitchBlock]:
        """Single letter inside a decorative circle."""
        blocks = self._single_letter(letter, size * 0.6, color)
        border = self._circle_border(size / 2, color, width=size * 0.06)
        if border.stitch_count > 0:
            blocks.append(border)
        return blocks

    def _badge(self, letter: str, size: float, color: str) -> List[StitchBlock]:
        """Letter inside a badge/shield shape."""
        blocks = self._single_letter(letter, size * 0.5, color)

        # Shield outline
        r = size / 2
        shield_points = [
            (-r, -r * 0.8), (0, -r), (r, -r * 0.8),
            (r, r * 0.2), (0, r), (-r, r * 0.2), (-r, -r * 0.8)
        ]
        block = self.satin_gen.generate(
            center_points=shield_points,
            width=size * 0.05, density=0.3,
            color=color, color_name="Thread",
            name="Badge Outline"
        )
        if block.stitch_count > 0:
            blocks.append(block)

        return blocks

    def _circle_border(self, radius: float, color: str, width: float = None) -> StitchBlock:
        """Generate a circular border."""
        points = []
        segments = 60
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle) * 0.9
            y = radius * math.sin(angle) * 0.9
            points.append((x, y))

        return self.satin_gen.generate(
            center_points=points,
            width=width or radius * 0.08,
            density=0.3,
            color=color,
            color_name="Thread",
            name="Circle Border"
        )
