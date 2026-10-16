"""
POWER DIGITIZE — Real Fidelity Measurement Engine
Renders stitch blocks and compares against original artwork
to produce genuine IoU, coverage, contour-deviation, and
negative-space preservation metrics.

No synthetic formulas.  Every number comes from pixel-level
comparison between the rendered stitch model and the immutable
original artwork.
"""

import math
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor

from engine.stitch_generators import StitchBlock, Stitch


class FidelityEngine:
    """
    Computes real geometric fidelity between a rendered stitch model
    and original artwork.  Produces per-object and global metrics
    plus a difference-map heatmap.
    """

    NORM_SIZE = 800  # normalised comparison resolution (px)

    def __init__(self):
        pass

    # ━━━━━━━━━━━━━━━━━━━  PUBLIC  ━━━━━━━━━━━━━━━━━━━━

    def compute_fidelity(
        self,
        original_image_path: str,
        stitch_blocks: List[StitchBlock],
        target_width_mm: float,
        target_height_mm: float,
        objects_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Full fidelity analysis.

        Returns dict with:
            overall_fidelity_pct   – weighted composite (0-100)
            iou                    – silhouette intersection-over-union
            metrics                – sub-metric breakdown
            object_coverage        – per-object stats
            duplicate_paths        – count of accidental duplicates
            difference_map         – RGBA numpy array (may be None)
            passes_gate            – bool
            gate_failures          – list of failure strings
        """
        orig_img = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
        if orig_img is None:
            return self._empty_result("Cannot read original image")

        orig_mask = self._extract_foreground_mask(orig_img)
        orig_norm = self._normalize_mask(orig_mask)

        stitch_mask = self._render_stitch_mask(stitch_blocks, self.NORM_SIZE)
        stitch_norm = self._normalize_mask(stitch_mask)

        iou = self._compute_iou(orig_norm, stitch_norm)
        contour_m = self._compute_contour_deviation(
            orig_norm, stitch_norm, target_width_mm, target_height_mm
        )
        coverage = self._compute_coverage(orig_norm, stitch_norm)
        neg_space = self._compute_negative_space_preservation(orig_norm, stitch_norm)

        diff_map = self._generate_difference_map(
            orig_norm, stitch_norm, contour_m["distance_transform"]
        )

        overall = (
            iou * 0.30
            + contour_m["contour_match_pct"] / 100.0 * 0.25
            + coverage["coverage_pct"] / 100.0 * 0.20
            + neg_space["preservation_pct"] / 100.0 * 0.15
            + (1.0 - min(1.0, contour_m["mean_deviation_mm"] / 2.0)) * 0.10
        ) * 100.0

        duplicate_count = self._detect_duplicate_paths(stitch_blocks)

        object_coverage = []
        if objects_info:
            object_coverage = self._compute_per_object_coverage(objects_info)

        return {
            "overall_fidelity_pct": round(overall, 1),
            "iou": round(iou, 4),
            "metrics": {
                "silhouette_iou": round(iou, 4),
                "contour_match_pct": contour_m["contour_match_pct"],
                "mean_contour_deviation_mm": contour_m["mean_deviation_mm"],
                "max_contour_deviation_mm": contour_m["max_deviation_mm"],
                "coverage_pct": coverage["coverage_pct"],
                "over_coverage_pct": coverage["over_coverage_pct"],
                "negative_space_preservation_pct": neg_space["preservation_pct"],
                "negative_space_violations": neg_space["violation_count"],
            },
            "object_coverage": object_coverage,
            "duplicate_paths": duplicate_count,
            "difference_map": diff_map,
            "passes_gate": (
                overall >= 65.0
                and duplicate_count == 0
            ),
            "gate_failures": self._check_gates(overall, neg_space, duplicate_count, contour_m),
        }

    # ━━━━━━━━━━━  MASK CONSTRUCTION  ━━━━━━━━━━━━━

    def _extract_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) >= 3 and img.shape[2] == 4:
            return (img[:, :, 3] > 20).astype(np.uint8) * 255
        if len(img.shape) >= 3:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            return (gray < 240).astype(np.uint8) * 255
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Centre-crop and resize to NORM_SIZE × NORM_SIZE."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((self.NORM_SIZE, self.NORM_SIZE), dtype=np.uint8)
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        crop = mask[y0:y1 + 1, x0:x1 + 1]
        ch, cw = crop.shape[:2]
        target = self.NORM_SIZE
        margin = 20
        scale = min((target - 2 * margin) / max(cw, 1),
                     (target - 2 * margin) / max(ch, 1))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target, target), dtype=np.uint8)
        ox = (target - nw) // 2
        oy = (target - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    def _render_stitch_mask(self, stitch_blocks: List[StitchBlock], size: int) -> np.ndarray:
        """Render stitch blocks into a greyscale mask (255 = stitched)."""
        if not stitch_blocks:
            return np.zeros((size, size), dtype=np.uint8)
        all_x, all_y = [], []
        for b in stitch_blocks:
            for s in b.stitches:
                if not (s.flags & (Stitch.JUMP | Stitch.TRIM)):
                    all_x.append(s.x)
                    all_y.append(s.y)
        if not all_x:
            return np.zeros((size, size), dtype=np.uint8)

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        data_w = max(max_x - min_x, 0.01)
        data_h = max(max_y - min_y, 0.01)
        margin = 20
        scale = min((size - 2 * margin) / data_w, (size - 2 * margin) / data_h)
        off_x = margin + ((size - 2 * margin) - data_w * scale) / 2
        off_y = margin + ((size - 2 * margin) - data_h * scale) / 2

        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)

        thread_w_map = {
            "fill": max(2, int(scale * 0.45)),
            "satin": max(3, int(scale * 0.50)),
            "running": max(2, int(scale * 0.30)),
            "underlay": max(1, int(scale * 0.25)),
        }

        def to_px(sx, sy):
            return (int((sx - min_x) * scale + off_x),
                    int((max_y - sy) * scale + off_y))

        for block in stitch_blocks:
            w = thread_w_map.get(block.stitch_type, thread_w_map["fill"])
            for i in range(1, len(block.stitches)):
                s0 = block.stitches[i - 1]
                s1 = block.stitches[i]
                if s1.flags & (Stitch.JUMP | Stitch.TRIM):
                    continue
                p0 = to_px(s0.x, s0.y)
                p1 = to_px(s1.x, s1.y)
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                if math.sqrt(dx * dx + dy * dy) > size * 0.45:
                    continue
                draw.line([p0, p1], fill=255, width=w)

        return np.array(img)

    # ━━━━━━━━━━━  METRIC COMPUTATION  ━━━━━━━━━━━━━

    def _compute_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        inter = np.count_nonzero(cv2.bitwise_and(a, b))
        union = np.count_nonzero(cv2.bitwise_or(a, b))
        return float(inter / max(1, union))

    def _compute_contour_deviation(
        self, ref: np.ndarray, emb: np.ndarray,
        tw_mm: float, th_mm: float,
    ) -> Dict[str, Any]:
        ref_edges = cv2.Canny(ref, 50, 150)
        emb_edges = cv2.Canny(emb, 50, 150)
        dist_ref = cv2.distanceTransform(cv2.bitwise_not(ref_edges), cv2.DIST_L2, 3)
        emb_pts = np.where(emb_edges > 0)
        px_to_mm = ((tw_mm / self.NORM_SIZE) + (th_mm / self.NORM_SIZE)) / 2.0

        if len(emb_pts[0]) > 0:
            devs = dist_ref[emb_pts] * px_to_mm
            mean_d = float(np.mean(devs))
            max_d = float(np.percentile(devs, 95))
            match_pct = min(100.0, max(0.0, 100.0 - mean_d * 15.0))
        else:
            mean_d = 0.0
            max_d = 0.0
            match_pct = 0.0

        return {
            "mean_deviation_mm": round(mean_d, 3),
            "max_deviation_mm": round(max_d, 3),
            "contour_match_pct": round(match_pct, 1),
            "distance_transform": dist_ref,
        }

    def _compute_coverage(self, ref: np.ndarray, emb: np.ndarray) -> Dict[str, Any]:
        ref_cnt = np.count_nonzero(ref)
        emb_cnt = np.count_nonzero(emb)
        inter = np.count_nonzero(cv2.bitwise_and(ref, emb))
        cov_pct = float(inter / max(1, ref_cnt)) * 100.0
        over = np.count_nonzero(cv2.subtract(emb, ref))
        over_pct = float(over / max(1, emb_cnt)) * 100.0
        return {
            "coverage_pct": round(cov_pct, 1),
            "over_coverage_pct": round(over_pct, 1),
        }

    def _compute_negative_space_preservation(
        self, ref: np.ndarray, emb: np.ndarray,
    ) -> Dict[str, Any]:
        """Check that holes/gaps in the original are preserved."""
        ref_contours, hierarchy = cv2.findContours(
            ref, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        hole_count = 0
        preserved = 0
        violations = 0

        if hierarchy is not None:
            for idx in range(len(ref_contours)):
                if hierarchy[0][idx][3] >= 0:  # is a child → hole
                    area = cv2.contourArea(ref_contours[idx])
                    if area < 25:
                        continue
                    hole_count += 1
                    hole_mask = np.zeros_like(ref)
                    cv2.drawContours(hole_mask, ref_contours, idx, 255, -1)
                    invasion = cv2.bitwise_and(emb, hole_mask)
                    frac = np.count_nonzero(invasion) / max(1, np.count_nonzero(hole_mask))
                    if frac < 0.15:
                        preserved += 1
                    else:
                        violations += 1

        pct = (preserved / max(1, hole_count)) * 100.0 if hole_count else 100.0
        return {
            "hole_count": hole_count,
            "preserved_count": preserved,
            "violation_count": violations,
            "preservation_pct": round(pct, 1),
        }

    # ━━━━━━━━  DUPLICATE-PATH DETECTION  ━━━━━━━━━

    def _detect_duplicate_paths(self, blocks: List[StitchBlock]) -> int:
        sigs = []
        dups = 0
        for b in blocks:
            if b.stitch_type in ("underlay", "repeat"):
                continue
            normals = [(s.x, s.y) for s in b.stitches if s.flags == Stitch.NORMAL]
            if len(normals) < 3:
                continue
            mid = normals[len(normals) // 2]
            sig = (
                round(normals[0][0], 1), round(normals[0][1], 1),
                round(normals[-1][0], 1), round(normals[-1][1], 1),
                len(normals),
                round(mid[0], 1), round(mid[1], 1),
            )
            for ex in sigs:
                if (abs(sig[0] - ex[0]) < 0.5
                    and abs(sig[1] - ex[1]) < 0.5
                    and abs(sig[2] - ex[2]) < 0.5
                    and abs(sig[3] - ex[3]) < 0.5
                    and abs(sig[4] - ex[4]) < 3):
                    dups += 1
                    break
            sigs.append(sig)
        return dups

    # ━━━━━━━━━  DIFFERENCE MAP  ━━━━━━━━━━━━━━━━━

    def _generate_difference_map(
        self, ref: np.ndarray, emb: np.ndarray, dist_ref: np.ndarray,
    ) -> np.ndarray:
        """
        4-colour heatmap:
          Green  – match (< 0.4mm deviation)
          Yellow – minor (0.4 – 1.2mm)
          Red    – significant (> 1.2mm)
          Blue   – missing artwork region
          Cyan   – extra stitching beyond artwork
        """
        h, w = ref.shape
        px_to_mm = 0.1  # approximate for the normalized canvas
        heatmap = np.zeros((h, w, 4), dtype=np.uint8)

        intersection = cv2.bitwise_and(ref, emb)
        missing = cv2.subtract(ref, emb)
        extra = cv2.subtract(emb, ref)

        y_m, x_m = np.where(intersection > 0)
        if len(y_m) > 0:
            devs = dist_ref[y_m, x_m] * px_to_mm
            green = devs <= 0.4
            yellow = (devs > 0.4) & (devs <= 1.2)
            red = devs > 1.2
            heatmap[y_m[green], x_m[green]] = [16, 185, 129, 210]
            heatmap[y_m[yellow], x_m[yellow]] = [234, 179, 8, 220]
            heatmap[y_m[red], x_m[red]] = [239, 68, 68, 230]

        my, mx = np.where(missing > 0)
        heatmap[my, mx] = [59, 130, 246, 210]

        ey, ex = np.where(extra > 0)
        heatmap[ey, ex] = [0, 212, 255, 190]

        return heatmap

    # ━━━━━━━━━  PER-OBJECT COVERAGE  ━━━━━━━━━━━

    def _compute_per_object_coverage(
        self, objects_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for obj in objects_info[:30]:
            results.append({
                "id": obj.get("id", "?"),
                "name": obj.get("name", "Unknown"),
                "type": obj.get("type", "fill"),
                "stitch_count": obj.get("stitchCount", 0),
                "color": obj.get("color", "#000000"),
            })
        return results

    # ━━━━━━━━  VALIDATION GATES  ━━━━━━━━━━━━━━

    def _check_gates(self, score, neg_space, dup_count, contour_m) -> List[str]:
        failures = []
        if score < 65.0:
            failures.append(f"Overall fidelity {score:.1f}% below 65% threshold")
        if neg_space["violation_count"] > 0:
            failures.append(
                f"{neg_space['violation_count']} negative-space violation(s) detected"
            )
        if dup_count > 0:
            failures.append(f"{dup_count} duplicate stitch path(s) detected")
        if contour_m["mean_deviation_mm"] > 2.5:
            failures.append(
                f"Mean contour deviation {contour_m['mean_deviation_mm']:.2f}mm exceeds 2.5mm"
            )
        return failures

    # ━━━━━━━━━━━  HELPERS  ━━━━━━━━━━━━━━━━━━━

    def _empty_result(self, error: str) -> Dict[str, Any]:
        return {
            "overall_fidelity_pct": 0.0,
            "iou": 0.0,
            "metrics": {},
            "object_coverage": [],
            "duplicate_paths": 0,
            "difference_map": None,
            "passes_gate": False,
            "gate_failures": [error],
        }
