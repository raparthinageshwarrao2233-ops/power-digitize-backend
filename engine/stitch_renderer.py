"""
POWER DIGITIZE — Stitch Renderer (v4 – Solid Fill Preview)
Renders stitch data as a high-quality preview image with realistic thread simulation.

Features:
- 4x supersampling for crisp anti-aliasing
- Thread widths sized to guarantee solid fill coverage (no dot gaps)
- 4-layer 3D thread: deep shadow + mid shadow + body + specular highlight
- Skips travel/jump stitches cleanly
- Fill/satin/running/underlay visual differentiation
"""

import math
from typing import List
from PIL import Image, ImageDraw, ImageColor
from engine.stitch_generators import StitchBlock, Stitch


class StitchRenderer:
    """Renders stitch blocks into a preview image."""

    @classmethod
    def render_preview(
        cls,
        stitch_blocks: List[StitchBlock],
        output_path: str,
        width: int = 800,
        height: int = 800,
        bg_color: tuple = (0, 0, 0, 0),
        show_jumps: bool = False,
        render_mode: str = "thread",
    ):
        """Render all stitch blocks to a PNG preview image."""
        # 4x supersampling — eliminates jagged edges completely
        SS = 4
        render_w = width * SS
        render_h = height * SS
        img = Image.new("RGBA", (render_w, render_h), bg_color)
        draw = ImageDraw.Draw(img)

        if not stitch_blocks:
            img.resize((width, height), Image.LANCZOS).save(output_path, format="PNG")
            return

        # Bounds — only use actual sewn stitches (ignore JUMP/TRIM for bounds)
        all_x, all_y = [], []
        for block in stitch_blocks:
            for s in block.stitches:
                if not (s.flags & (Stitch.JUMP | Stitch.TRIM | Stitch.COLOR_CHANGE)):
                    all_x.append(s.x)
                    all_y.append(s.y)

        if not all_x:
            img.resize((width, height), Image.LANCZOS).save(output_path, format="PNG")
            return

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        data_w = max_x - min_x
        data_h = max_y - min_y

        if data_w < 0.01 or data_h < 0.01:
            img.resize((width, height), Image.LANCZOS).save(output_path, format="PNG")
            return

        # Scale: leave margin on each side
        margin = 80 * SS // 2
        scale_x = (render_w - 2 * margin) / data_w
        scale_y = (render_h - 2 * margin) / data_h
        scale = min(scale_x, scale_y)

        off_x = margin + ((render_w - 2 * margin) - data_w * scale) / 2
        off_y = margin + ((render_h - 2 * margin) - data_h * scale) / 2

        # --- Thread widths ---
        # Key insight: fill row spacing is 0.25-0.40mm.
        # Thread must be WIDER than row spacing to guarantee solid coverage.
        # At 4x SS, scale is already 4x larger, so the formula works directly.
        fill_w  = max(SS * 4, int(scale * 0.52))   # >density for overlap
        satin_w = max(SS * 5, int(scale * 0.58))
        run_w   = max(SS * 3, int(scale * 0.35))
        und_w   = max(SS * 2, int(scale * 0.28))

        flat_fill_w  = max(SS * 3, int(scale * 0.48))
        flat_satin_w = max(SS * 4, int(scale * 0.54))
        flat_run_w   = max(SS * 2, int(scale * 0.32))
        flat_und_w   = max(SS * 2, int(scale * 0.24))

        def to_px(x, y):
            return (
                int((x - min_x) * scale + off_x),
                int((max_y - y) * scale + off_y),   # flip Y: embroidery +Y is up
            )

        # Draw each block
        for block in stitch_blocks:
            color   = block.color
            stitches = block.stitches
            st_type  = block.stitch_type

            if render_mode == "thread":
                if st_type == "satin":
                    base_w = satin_w
                elif st_type == "underlay":
                    base_w = und_w
                elif st_type in ("running",):
                    base_w = run_w
                else:
                    base_w = fill_w
            else:
                if st_type == "satin":
                    base_w = flat_satin_w
                elif st_type == "underlay":
                    base_w = flat_und_w
                elif st_type in ("running",):
                    base_w = flat_run_w
                else:
                    base_w = flat_fill_w

            try:
                red, green, blue = ImageColor.getrgb(color)[:3]
            except Exception:
                red, green, blue = 190, 190, 190

            r = base_w // 2

            # Filter and collect valid stitch segments
            segs = []
            for i in range(1, len(stitches)):
                s0 = stitches[i - 1]
                s1 = stitches[i]

                # JUMP/TRIM handling
                if s1.flags & Stitch.JUMP:
                    if show_jumps:
                        draw.line(
                            [to_px(s0.x, s0.y), to_px(s1.x, s1.y)],
                            fill="#ffffff18", width=max(1, SS // 2)
                        )
                    continue
                if s1.flags & (Stitch.TRIM | Stitch.COLOR_CHANGE):
                    continue

                p0 = to_px(s0.x, s0.y)
                p1 = to_px(s1.x, s1.y)

                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                seg_px = math.hypot(dx, dy)
                if seg_px > render_w * 0.45:
                    continue

                segs.append((p0, p1, dx, dy, seg_px))

            if not segs:
                continue

            if render_mode == "thread":
                # ── Physical 4-Pass 3D Thread Simulation ──
                # Prevents painter's algorithm highlight occlusion between scanlines

                # Pass 1: Under-thread contact shadows (subtle ambient occlusion)
                s_r = max(0, int(red * 0.25))
                s_g = max(0, int(green * 0.25))
                s_b = max(0, int(blue * 0.25))
                for p0, p1, dx, dy, seg_len in segs:
                    draw.line([(p0[0] + 1, p0[1] + 1), (p1[0] + 1, p1[1] + 1)],
                              fill=(s_r, s_g, s_b, 160), width=base_w)

                # Pass 2: Main thread cylinder body with rounded endcaps
                for p0, p1, dx, dy, seg_len in segs:
                    draw.line([p0, p1], fill=(red, green, blue, 255), width=base_w)
                    if r > 0:
                        draw.ellipse([p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r],
                                     fill=(red, green, blue, 255))
                        draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r],
                                     fill=(red, green, blue, 255))

                # Pass 3: Needle penetration dimples (skip for underlays)
                if st_type != "underlay":
                    punc_r = max(1, int(r * 0.55))
                    for p0, p1, dx, dy, seg_len in segs:
                        draw.ellipse([p0[0] - punc_r, p0[1] - punc_r, p0[0] + punc_r, p0[1] + punc_r],
                                     fill=(s_r, s_g, s_b, 175))
                        draw.ellipse([p1[0] - punc_r, p1[1] - punc_r, p1[0] + punc_r, p1[1] + punc_r],
                                     fill=(s_r, s_g, s_b, 175))

                # Pass 4: Directional specular crest highlights (skip for underlays)
                # Drawn on top of all thread bodies so highlights are never obscured by traversal order
                if st_type != "underlay":
                    for p0, p1, dx, dy, seg_len in segs:
                        if seg_len > 2.0 and base_w >= 3:
                            nx = -dy / seg_len
                            ny = dx / seg_len
                            if nx * (-0.707) + ny * (-0.707) < 0:
                                nx, ny = -nx, -ny

                            offset_dist = max(0.5, base_w * 0.18)
                            hx0 = int(p0[0] + nx * offset_dist)
                            hy0 = int(p0[1] + ny * offset_dist)
                            hx1 = int(p1[0] + nx * offset_dist)
                            hy1 = int(p1[1] + ny * offset_dist)

                            light_dot = abs(nx * (-0.707) + ny * (-0.707))
                            sheen = 0.20 + 0.30 * light_dot
                            # Specular sheen retains dye saturation instead of washing out to chalky pink/white
                            hi_r = min(255, int(red + (255 - red) * sheen * 0.40))
                            hi_g = min(255, int(green + (255 - green) * sheen * 0.18))
                            hi_b = min(255, int(blue + (255 - blue) * sheen * 0.18))
                            hi_w = max(1, int(base_w * 0.22))
                            draw.line([(hx0, hy0), (hx1, hy1)], fill=(hi_r, hi_g, hi_b, 175), width=hi_w)

            else:
                # ── Flat / Diagnostic Mode ──
                # Pure thread geometry and exact assigned colors with zero lighting artifacts
                for p0, p1, dx, dy, seg_len in segs:
                    draw.line([p0, p1], fill=(red, green, blue, 255), width=base_w)
                    if r > 0:
                        draw.ellipse([p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r],
                                     fill=(red, green, blue, 255))
                        draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r],
                                     fill=(red, green, blue, 255))

        # Downscale with LANCZOS anti-aliasing
        final = img.resize((width, height), Image.LANCZOS)
        final.save(output_path, format="PNG")

    @classmethod
    def render_stitch_data_json(
        cls, stitch_blocks: List[StitchBlock]
    ) -> dict:
        """Convert stitch data to JSON for frontend rendering."""
        result = {
            "blocks": [],
            "bounds": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
        }

        all_x, all_y = [], []

        for block in stitch_blocks:
            block_data = {
                "color": block.color,
                "colorName": block.color_name,
                "stitchType": block.stitch_type,
                "name": block.name,
                "stitches": [],
            }

            for s in block.stitches:
                all_x.append(s.x)
                all_y.append(s.y)
                block_data["stitches"].append({
                    "x": round(s.x, 2),
                    "y": round(s.y, 2),
                    "flags": s.flags,
                })

            result["blocks"].append(block_data)

        if all_x:
            result["bounds"] = {
                "min_x": round(min(all_x), 2),
                "min_y": round(min(all_y), 2),
                "max_x": round(max(all_x), 2),
                "max_y": round(max(all_y), 2),
            }

        return result
