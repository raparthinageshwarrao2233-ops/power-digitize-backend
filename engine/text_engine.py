"""
POWER DIGITIZE — Text Embroidery Engine
Generates embroidery stitch data from text input with multi-font and arc support.
"""

import math
from typing import Dict, Any, List, Tuple
from engine.stitch_generators import StitchBlock, SatinStitchGenerator, RunningStitchGenerator, FillStitchGenerator


# Standard Stroke Font (Clean Block Sans)
# 10 wide x 14 tall normalized units
STROKE_FONT = {
    'A': [[(0,14),(5,0),(10,14)], [(2.5,8),(7.5,8)]],
    'B': [[(0,14),(0,0),(7,0),(9,2),(9,5),(7,7),(0,7)], [(7,7),(9,9),(9,12),(7,14),(0,14)]],
    'C': [[(10,2),(8,0),(2,0),(0,2),(0,12),(2,14),(8,14),(10,12)]],
    'D': [[(0,0),(0,14),(6,14),(10,10),(10,4),(6,0),(0,0)]],
    'E': [[(10,0),(0,0),(0,14),(10,14)], [(0,7),(7,7)]],
    'F': [[(10,0),(0,0),(0,14)], [(0,7),(7,7)]],
    'G': [[(10,2),(8,0),(2,0),(0,2),(0,12),(2,14),(8,14),(10,12),(10,7),(6,7)]],
    'H': [[(0,0),(0,14)], [(10,0),(10,14)], [(0,7),(10,7)]],
    'I': [[(3,0),(7,0)], [(5,0),(5,14)], [(3,14),(7,14)]],
    'J': [[(3,0),(7,0)], [(5,0),(5,12),(3,14),(1,14),(0,12)]],
    'K': [[(0,0),(0,14)], [(10,0),(0,7),(10,14)]],
    'L': [[(0,0),(0,14),(10,14)]],
    'M': [[(0,14),(0,0),(5,7),(10,0),(10,14)]],
    'N': [[(0,14),(0,0),(10,14),(10,0)]],
    'O': [[(2,0),(0,2),(0,12),(2,14),(8,14),(10,12),(10,2),(8,0),(2,0)]],
    'P': [[(0,14),(0,0),(7,0),(10,2),(10,5),(7,7),(0,7)]],
    'Q': [[(2,0),(0,2),(0,12),(2,14),(8,14),(10,12),(10,2),(8,0),(2,0)], [(7,11),(10,14)]],
    'R': [[(0,14),(0,0),(7,0),(10,2),(10,5),(7,7),(0,7)], [(6,7),(10,14)]],
    'S': [[(10,2),(8,0),(2,0),(0,2),(0,5),(2,7),(8,7),(10,9),(10,12),(8,14),(2,14),(0,12)]],
    'T': [[(0,0),(10,0)], [(5,0),(5,14)]],
    'U': [[(0,0),(0,12),(2,14),(8,14),(10,12),(10,0)]],
    'V': [[(0,0),(5,14),(10,0)]],
    'W': [[(0,0),(2.5,14),(5,7),(7.5,14),(10,0)]],
    'X': [[(0,0),(10,14)], [(10,0),(0,14)]],
    'Y': [[(0,0),(5,7),(10,0)], [(5,7),(5,14)]],
    'Z': [[(0,0),(10,0),(0,14),(10,14)]],
    '0': [[(2,0),(0,2),(0,12),(2,14),(8,14),(10,12),(10,2),(8,0),(2,0)]],
    '1': [[(3,2),(5,0),(5,14)], [(3,14),(7,14)]],
    '2': [[(0,2),(2,0),(8,0),(10,2),(10,5),(0,14),(10,14)]],
    '3': [[(0,2),(2,0),(8,0),(10,2),(10,5),(7,7),(10,9),(10,12),(8,14),(2,14),(0,12)]],
    '4': [[(8,14),(8,0),(0,10),(10,10)]],
    '5': [[(10,0),(0,0),(0,7),(8,7),(10,9),(10,12),(8,14),(2,14),(0,12)]],
    '6': [[(8,0),(2,0),(0,2),(0,12),(2,14),(8,14),(10,12),(10,9),(8,7),(0,7)]],
    '7': [[(0,0),(10,0),(4,14)]],
    '8': [[(2,0),(0,2),(0,5),(2,7),(8,7),(10,9),(10,12),(8,14),(2,14),(0,12),(0,9),(2,7)], [(8,7),(10,5),(10,2),(8,0),(2,0)]],
    '9': [[(10,7),(2,7),(0,5),(0,2),(2,0),(8,0),(10,2),(10,12),(8,14),(2,14)]],
    ' ': [],
    '.': [[(4,13),(4,14),(6,14),(6,13),(4,13)]],
    ',': [[(5,12),(5,14),(3,16)]],
    '-': [[(2,7),(8,7)]],
    '+': [[(5,3),(5,11)], [(1,7),(9,7)]],
    '!': [[(5,0),(5,10)], [(5,13),(5,14)]],
    '?': [[(0,2),(2,0),(8,0),(10,2),(10,5),(5,7),(5,10)], [(5,13),(5,14)]],
    '@': [[(8,5),(6,5),(5,6),(5,9),(6,10),(8,10),(9,8),(9,3),(7,1),(3,1),(1,3),(1,11),(3,13),(7,13),(9,11)]],
    '&': [[(10,14),(2,5),(4,2),(5,0),(6,0),(7,1),(7,3),(0,10),(0,12),(2,14),(5,14),(10,9)]],
    '#': [[(3,0),(1,14)], [(7,0),(5,14)], [(0,4),(8,4)], [(0,10),(8,10)]],
    '*': [[(5,3),(5,11)], [(1.5,4.5),(8.5,9.5)], [(1.5,9.5),(8.5,4.5)]],
}


class TextEmbroideryEngine:
    """Generate embroidery from text input with font family & curvature support."""

    def __init__(self):
        self.satin_gen = SatinStitchGenerator()
        self.running_gen = RunningStitchGenerator()
        self.fill_gen = FillStitchGenerator()

    def _get_strokes_for_char(self, char: str, font_family: str) -> List[List[Tuple[float, float]]]:
        base_strokes = STROKE_FONT.get(char, [])
        if not base_strokes:
            return []

        if font_family == "athletic":
            chamfers = {
                'O': [[(3,0),(7,0),(10,3),(10,11),(7,14),(3,14),(0,11),(0,3),(3,0)]],
                'C': [[(10,3),(7,0),(3,0),(0,3),(0,11),(3,14),(7,14),(10,11)]],
                'D': [[(0,0),(0,14),(7,14),(10,11),(10,3),(7,0),(0,0)]],
                'G': [[(10,3),(7,0),(3,0),(0,3),(0,11),(3,14),(7,14),(10,11),(10,7),(6,7)]],
                '0': [[(3,0),(7,0),(10,3),(10,11),(7,14),(3,14),(0,11),(0,3),(3,0)]],
                '8': [[(3,0),(7,0),(10,3),(10,5),(7,7),(3,7),(0,5),(0,3),(3,0)],
                      [(3,7),(7,7),(10,9),(10,11),(7,14),(3,14),(0,11),(0,9),(3,7)]],
            }
            return chamfers.get(char, base_strokes)

        elif font_family == "serif":
            feet = {
                'I': [[(1,0),(9,0)], [(5,0),(5,14)], [(1,14),(9,14)]],
                'H': [[(0,0),(0,14)], [(10,0),(10,14)], [(0,7),(10,7)], [(-1.5,0),(1.5,0)], [(8.5,0),(11.5,0)], [(-1.5,14),(1.5,14)], [(8.5,14),(11.5,14)]],
                'A': [[(0,14),(5,0),(10,14)], [(2.5,8),(7.5,8)], [(-1.5,14),(1.5,14)], [(8.5,14),(11.5,14)]],
                'T': [[(-1,0),(11,0)], [(5,0),(5,14)], [(3,14),(7,14)]],
                'L': [[(0,0),(0,14),(10,14)], [(-1.5,0),(1.5,0)], [(10,12),(10,14)]],
            }
            return feet.get(char, base_strokes)

        elif font_family == "script":
            # Slanted forward italic with loop flair
            res = []
            for stroke in base_strokes:
                new_s = []
                for x, y in stroke:
                    slanted_x = x + (14.0 - y) * 0.22
                    new_s.append((slanted_x, y))
                res.append(new_s)
            return res

        elif font_family == "stencil":
            stencils = {
                'O': [[(2,0),(0,2),(0,6)], [(0,8),(0,12),(2,14),(8,14),(10,12),(10,8)], [(10,6),(10,2),(8,0),(2,0)]],
                'D': [[(0,0),(0,6)], [(0,8),(0,14),(6,14),(10,10),(10,4),(6,0),(0,0)]],
                'A': [[(0,14),(3,5)], [(4,2),(5,0),(6,2)], [(7,5),(10,14)], [(2.5,8),(7.5,8)]],
                'B': [[(0,0),(0,6)], [(0,8),(0,14),(7,14),(9,12),(9,9),(7,7)], [(0,7),(7,7),(9,5),(9,2),(7,0),(0,0)]],
            }
            return stencils.get(char, base_strokes)

        return base_strokes

    def _arc_transform(self, points: List[Tuple[float, float]], arc_deg: float, total_width: float) -> List[Tuple[float, float]]:
        if abs(arc_deg) < 1.0 or total_width <= 0:
            return points
        theta_rad = math.radians(abs(arc_deg))
        R = total_width / theta_rad
        sign = 1.0 if arc_deg > 0 else -1.0
        transformed = []
        for x, y in points:
            angle = (x / total_width) * theta_rad * sign
            cur_R = R - y * sign
            new_x = cur_R * math.sin(angle)
            new_y = (R - cur_R * math.cos(angle)) * sign
            transformed.append((new_x, new_y))
        return transformed

    def generate(
        self,
        text: str,
        design_id: str,
        font_size_mm: float = 15.0,
        stitch_type: str = "satin",
        color: str = "#facf3b",
        color_mode: str = "single",  # 'single' | 'varsity_outline' | 'alternating_words' | 'alternating_chars'
        secondary_color: str = "#00d4ff",
        palette: list = None,
        alignment: str = "center",
        style: str = "normal",
        font_family: str = "athletic",
        arc_angle: float = 0.0,
        spacing: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate professional Tajima DST embroidery stitch data with multi-color and varsity support."""

        scale = max(0.2, font_size_mm / 14.0)
        char_width = 10 * scale
        char_spacing = 2.5 * scale * max(0.1, spacing)

        total_width = len(text) * char_width + max(0, len(text) - 1) * char_spacing

        # Base horizontal starting offset relative to text box
        if alignment == "center":
            start_x = -total_width / 2.0
        elif alignment == "right":
            start_x = -total_width
        else:
            start_x = 0.0

        start_y = 0.0
        palette = palette or [color, secondary_color, "#e11d48", "#ffffff", "#10b981", "#f97316"]

        # Solid commercial satin column width (22% of font size, min 1.2mm, max 8.5mm)
        satin_width = max(1.2, min(8.5, font_size_mm * 0.22))

        # First pass: collect transformed stroke paths for all characters
        char_data = []
        cursor_x = start_x
        words = text.split(" ")
        word_idx = 0

        for i, char in enumerate(text.upper()):
            if char == ' ':
                cursor_x += char_width * 0.8
                word_idx += 1
                continue

            # Determine character color
            if color_mode == "alternating_words":
                c = palette[word_idx % len(palette)]
            elif color_mode == "alternating_chars":
                c = palette[i % len(palette)]
            else:
                c = color

            strokes = self._get_strokes_for_char(char, font_family)
            char_strokes = []

            for stroke in strokes:
                # 1. Scale and position (+Y up in embroidery Cartesian)
                points = [
                    (p[0] * scale + cursor_x, (14.0 - p[1]) * scale + start_y)
                    for p in stroke
                ]
                if len(points) < 2:
                    continue

                # 2. Apply Arc curvature
                if abs(arc_angle) >= 1.0:
                    points = self._arc_transform(points, arc_angle, total_width)

                # 3. Apply global offset
                points = [(p[0] + offset_x, p[1] + offset_y) for p in points]

                # 4. Smooth corners for continuous satin column
                points = self._smooth_corners(points)
                char_strokes.append(points)

            char_data.append((char, char_strokes, c))
            cursor_x += char_width + char_spacing

        blocks = []
        objects_info = []

        if color_mode == "varsity_outline":
            # Commercial 2-color sequence:
            # Pass 1: Outline / Border layer in secondary_color (wider stroke)
            for idx, (char, strokes, _) in enumerate(char_data):
                for s_idx, points in enumerate(strokes):
                    b_outline = self.satin_gen.generate(
                        center_points=points,
                        width=satin_width * 1.5,
                        density=0.38,
                        pull_compensation=0.25,
                        underlay="contour",
                        color=secondary_color,
                        color_name="Varsity Outline",
                        name=f"Outline '{char}' #{s_idx+1}",
                    )
                    if b_outline.stitch_count > 0:
                        blocks.append(b_outline)
                        objects_info.append({
                            "id": f"varsity-outline-{idx}-{s_idx}",
                            "type": "satin",
                            "name": f"Varsity Outline '{char}'",
                            "color": secondary_color,
                            "colorName": "Varsity Outline",
                            "stitchCount": b_outline.stitch_count,
                            "visible": True,
                            "locked": False,
                        })

            # Pass 2: Inner core body in primary color (anchored with center underlay)
            for idx, (char, strokes, _) in enumerate(char_data):
                for s_idx, points in enumerate(strokes):
                    b_core = self.satin_gen.generate(
                        center_points=points,
                        width=satin_width * 0.85,
                        density=0.35,
                        pull_compensation=0.2,
                        underlay="center",
                        color=color,
                        color_name="Varsity Core",
                        name=f"Core '{char}' #{s_idx+1}",
                    )
                    if b_core.stitch_count > 0:
                        blocks.append(b_core)
                        objects_info.append({
                            "id": f"varsity-core-{idx}-{s_idx}",
                            "type": "satin",
                            "name": f"Varsity Core '{char}'",
                            "color": color,
                            "colorName": "Varsity Core",
                            "stitchCount": b_core.stitch_count,
                            "visible": True,
                            "locked": False,
                        })
        else:
            # Single Color or Alternating Palette
            for idx, (char, strokes, char_color) in enumerate(char_data):
                for s_idx, points in enumerate(strokes):
                    if stitch_type == "satin":
                        block = self.satin_gen.generate(
                            center_points=points,
                            width=satin_width,
                            density=0.35,
                            pull_compensation=0.25,
                            underlay="center",
                            color=char_color,
                            color_name="TextThread",
                            name=f"Letter '{char}' #{s_idx+1}",
                        )
                    else:
                        block = self.running_gen.generate(
                            points=points,
                            stitch_length=2.0,
                            color=char_color,
                            color_name="TextThread",
                            name=f"Letter '{char}' #{s_idx+1}",
                        )

                    if block.stitch_count > 0:
                        blocks.append(block)
                        objects_info.append({
                            "id": f"text-layer-{idx}-{s_idx}",
                            "type": stitch_type,
                            "name": f"Letter '{char}'",
                            "color": char_color,
                            "colorName": "TextThread",
                            "stitchCount": block.stitch_count,
                            "visible": True,
                            "locked": False,
                        })

        total_stitches = sum(b.stitch_count for b in blocks)
        thread_length = self._calc_thread_length(blocks)
        transitions = max(0, len(blocks) - 1)

        # Collect unique color segments in order of appearance
        ordered_colors = []
        for b in blocks:
            if b.color and b.color.lower() not in [c.lower() for c in ordered_colors]:
                ordered_colors.append(b.color)

        all_pts_x = [s.x for b in blocks for s in b.stitches]
        all_pts_y = [s.y for b in blocks for s in b.stitches]
        w_mm = (max(all_pts_x) - min(all_pts_x)) if all_pts_x else total_width
        h_mm = (max(all_pts_y) - min(all_pts_y)) if all_pts_y else font_size_mm

        return {
            "stitch_data": blocks,
            "objects": objects_info,
            "colors": [{"hex": c, "name": "Thread"} for c in ordered_colors],
            "stats": {
                "stitch_count": total_stitches,
                "jump_count": transitions,
                "trim_count": transitions,
                "color_count": max(1, len(ordered_colors)),
                "color_segments": ordered_colors,
                "width": round(w_mm, 1),
                "height": round(h_mm, 1),
                "thread_length": round(thread_length / 1000.0, 1),
                "estimated_time": max(1, round(total_stitches / 800)),
            },
        }


    def _calc_thread_length(self, blocks) -> float:
        total = 0.0
        for block in blocks:
            for i in range(1, len(block.stitches)):
                s0 = block.stitches[i-1]
                s1 = block.stitches[i]
                dx = s1.x - s0.x
                dy = s1.y - s0.y
                total += math.sqrt(dx*dx + dy*dy)
        return total

    def _smooth_corners(self, points: List[tuple]) -> List[tuple]:
        if len(points) < 3:
            return points
        smoothed = [points[0]]
        for i in range(1, len(points) - 1):
            p0, p1, p2 = points[i-1], points[i], points[i+1]
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if len1 == 0 or len2 == 0:
                smoothed.append(p1)
                continue
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            cos_a = max(-1.0, min(1.0, dot / (len1 * len2)))
            angle = math.degrees(math.acos(cos_a))
            if angle < 45.0:
                d1 = min(len1 * 0.2, 1.0)
                d2 = min(len2 * 0.2, 1.0)
                new_p1 = (p1[0] + (v1[0]/len1)*d1, p1[1] + (v1[1]/len1)*d1)
                new_p2 = (p1[0] + (v2[0]/len2)*d2, p1[1] + (v2[1]/len2)*d2)
                smoothed.append(new_p1)
                smoothed.append(new_p2)
            else:
                smoothed.append(p1)
        smoothed.append(points[-1])
        return smoothed
