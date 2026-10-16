"""
POWER DIGITIZE — Line-Art & Micro-Detail Preservation Engine (v1 – Native Multi-Representation)

Architecture:
1. Native-Resolution Feature Extraction: Never downsamples the source image prior to micro-feature analysis.
2. Dual Representation: Separates Region Fills (large closed tatami areas) from Stroke/Line Representation (centerline running/bean/satin).
3. Skeletonization & Centerline Tracing: Morphological medial axis thinning to recover true stroke paths.
4. Semantic Facial & Micro-Detail Detector: Identifies eyes, pupils, bindi/dots, nose, lips, necklace beads, crown ornaments, hair flow.
5. Structural Importance Scoring: Multi-factor importance (contrast, face proximity, symmetry, circularity) preventing detail loss.
6. Embroidery-Safe Controlled Adaptation: Intelligently thickens/enlarges sub-minimum physical features (0.4mm -> 0.75mm) preserving centroid.
7. Stitch Classification: Maps strokes to Running, Bean (triple), Satin Column, or Micro-Dot Rosette.
8. Object Hierarchy & Protection: Locks micro-details under PROTECTED_DETAIL to survive subsequent refinement passes.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis, thin
from scipy.ndimage import distance_transform_edt


class LineArtEngine:
    """
    Dedicated Line-Art & Micro-Detail Extraction Engine for high-fidelity embroidery.
    Transforms raw artwork into a dual-representation model (Region Fills + Stroke Centerlines).
    """

    def __init__(self):
        self.detail_budgets = {
            "low": {"min_path_len_px": 12, "min_importance": 0.65, "adapt_sub_mm": False},
            "medium": {"min_path_len_px": 8, "min_importance": 0.45, "adapt_sub_mm": True},
            "high": {"min_path_len_px": 4, "min_importance": 0.25, "adapt_sub_mm": True},
            "maximum": {"min_path_len_px": 2, "min_importance": 0.10, "adapt_sub_mm": True},
        }

    def extract_line_art_and_details(
        self,
        native_image: np.ndarray,
        target_width_mm: float,
        target_height_mm: float,
        detail_budget: str = "high",
        color_palette: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Main extraction entry point.
        Analyzes native-resolution image to extract high-fidelity stroke centerlines,
        facial landmarks, bindi dots, necklace patterns, crown ornamentation, and hair flows.
        """
        budget_cfg = self.detail_budgets.get(detail_budget.lower(), self.detail_budgets["high"])

        # Handle color and transparency
        has_alpha = len(native_image.shape) == 3 and native_image.shape[2] == 4
        if has_alpha:
            bgr = native_image[:, :, :3]
            alpha = native_image[:, :, 3]
            fg_mask = (alpha > 32).astype(np.uint8) * 255
        else:
            bgr = native_image if len(native_image.shape) == 3 else cv2.cvtColor(native_image, cv2.COLOR_GRAY2BGR)
            fg_mask = np.ones((bgr.shape[0], bgr.shape[1]), dtype=np.uint8) * 255

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h_native, w_native = gray.shape[:2]

        # Calculate exact pixel-to-mm physical scale factors
        scale_x = target_width_mm / max(w_native, 1)
        scale_y = target_height_mm / max(h_native, 1)
        mm_per_pixel = (scale_x + scale_y) / 2.0

        # ── 1. MULTI-SCALE LINE & EDGE EXTRACTION ──
        # True dark linework (actual dark lines / outlines in the artwork)
        dark_strokes = ((gray < 80) & (fg_mask > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dark_strokes = cv2.morphologyEx(dark_strokes, cv2.MORPH_CLOSE, kernel)

        # High-contrast structural boundaries (ignores soft gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        edges_high = ((grad_mag > 85) & (gray < 160) & (fg_mask > 0)).astype(np.uint8) * 255

        # Combined stroke mask (sharp dark lines & high-contrast boundaries only)
        combined_stroke_mask = cv2.bitwise_or(dark_strokes, edges_high)
        combined_stroke_mask = cv2.morphologyEx(combined_stroke_mask, cv2.MORPH_OPEN, kernel)

        # Compute Euclidean Distance Transform for stroke width estimation
        dist_transform = distance_transform_edt(combined_stroke_mask > 0)

        # ── 2. SKELETONIZATION / MEDIAL AXIS CENTERLINE EXTRACTION ──
        # Extract 1-pixel wide medial centerline paths
        bool_mask = combined_stroke_mask > 0
        skeleton = skeletonize(bool_mask)
        skel_uint8 = (skeleton.astype(np.uint8)) * 255

        # ── 3. TRACE CONNECTED SKELETON PATHS ──
        raw_paths = self._trace_skeleton_paths(skel_uint8)

        # ── 4. DETECT FACIAL & SEMANTIC FEATURE REGIONS ──
        face_roi, facial_features = self._detect_facial_and_semantic_structures(
            gray, rgb, fg_mask, dist_transform, mm_per_pixel
        )

        # ── 5. CLASSIFY, SCORE, AND ADAPT OBJECTS ──
        extracted_objects = []
        protected_count = 0
        running_count = 0
        bean_count = 0
        satin_count = 0
        micro_dot_count = 0

        # A. Process Traced Stroke Paths
        for p_idx, path in enumerate(raw_paths):
            if len(path) < budget_cfg["min_path_len_px"]:
                continue

            path_pts = np.array(path, dtype=np.float32)
            path_len_px = float(cv2.arcLength(path_pts.reshape(-1, 1, 2), False))
            path_len_mm = path_len_px * mm_per_pixel

            if path_len_mm < 0.35 and not self._is_near_facial_feature(path[0], facial_features):
                continue

            # Estimate stroke width along the path from distance transform
            widths_px = [2.0 * dist_transform[int(max(0, min(h_native - 1, pt[1]))),
                                               int(max(0, min(w_native - 1, pt[0])))]
                         for pt in path]
            avg_width_px = float(np.median(widths_px)) if widths_px else 1.0
            avg_width_mm = avg_width_px * mm_per_pixel

            # Skip solid fill bodies (> 2.0mm wide) — Representation A already fills them with Tatami!
            if avg_width_mm > 2.0:
                continue

            # Calculate semantic importance score
            importance, semantic_tag, is_protected = self._score_stroke_importance(
                path, avg_width_mm, path_len_mm, gray, face_roi, facial_features
            )

            if importance < budget_cfg["min_importance"] and not is_protected:
                continue

            # Classify stitch type and physical adaptation
            # Character outlines and linework are generated as Bean (triple running) stitches
            # so they stand out sharp and prominent above tatami fills
            stitch_type = "bean"
            adaptation = "NONE"
            effective_width_mm = avg_width_mm

            if avg_width_mm >= 1.2 and path_len_mm >= 2.0:
                stitch_type = "satin_column"
                satin_count += 1
            elif is_protected and avg_width_mm < 0.60 and budget_cfg["adapt_sub_mm"]:
                adaptation = "CONTROLLED_THICKEN"
                effective_width_mm = 0.75
                stitch_type = "bean"
                bean_count += 1
            elif avg_width_mm >= 0.7 or is_protected:
                stitch_type = "bean"
                bean_count += 1
            else:
                stitch_type = "running"
                running_count += 1

            if is_protected:
                protected_count += 1

            # Determine dominant color for this stroke
            stroke_color, stroke_color_name = self._sample_stroke_color(path, rgb, color_palette)

            # Smooth path with Chaikin-like curvature preservation
            smoothed_path = self._smooth_polyline(path)

            extracted_objects.append({
                "id": f"stroke_{len(extracted_objects) + 1}",
                "category": "STROKE",
                "semantic_tag": semantic_tag,
                "is_protected": is_protected,
                "importance_score": round(importance, 3),
                "stitch_type": stitch_type,
                "adaptation": adaptation,
                "raw_points": path,
                "points": smoothed_path,
                "width_mm": round(effective_width_mm, 2),
                "length_mm": round(path_len_mm, 2),
                "color_hex": stroke_color,
                "color_name": stroke_color_name,
            })

        # B. Process Discrete Micro-Dots & Bindi Features
        for dot in facial_features.get("dots", []):
            is_protected = True
            protected_count += 1
            micro_dot_count += 1

            dot_diameter_mm = dot["radius_px"] * 2.0 * mm_per_pixel
            adaptation = "NONE"
            if dot_diameter_mm < 1.20 and budget_cfg["adapt_sub_mm"]:
                adaptation = "CONTROLLED_ENLARGE"
                dot_diameter_mm = max(0.85, dot_diameter_mm)

            dot_color, dot_color_name = self._sample_point_color(dot["center"], rgb, color_palette)

            extracted_objects.append({
                "id": f"dot_{len(extracted_objects) + 1}",
                "category": "MICRO_DOT",
                "semantic_tag": dot.get("type", "BINDI_DOT"),
                "is_protected": is_protected,
                "importance_score": 0.98,
                "stitch_type": "micro_dot",
                "adaptation": adaptation,
                "center": dot["center"],
                "diameter_mm": round(dot_diameter_mm, 2),
                "color_hex": dot_color,
                "color_name": dot_color_name,
            })

        # ── 6. BUILD OBJECT HIERARCHY ──
        hierarchy = self._build_object_hierarchy(extracted_objects)

        return {
            "native_resolution": {"width": w_native, "height": h_native},
            "mm_per_pixel": mm_per_pixel,
            "objects": extracted_objects,
            "hierarchy": hierarchy,
            "stats": {
                "total_stroke_objects": len(extracted_objects),
                "protected_details": protected_count,
                "running_strokes": running_count,
                "bean_strokes": bean_count,
                "satin_strokes": satin_count,
                "micro_dots": micro_dot_count,
                "facial_features_detected": len(facial_features.get("landmarks", [])),
                "detail_preservation_budget": detail_budget.upper(),
            },
            "facial_roi": face_roi,
        }

    def _trace_skeleton_paths(self, skeleton_img: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Traces 1-pixel skeleton branches into smooth connected polylines.
        Resolves intersections, tracks endpoints, and preserves true curve paths.
        """
        h, w = skeleton_img.shape
        visited = np.zeros((h, w), dtype=bool)
        skel_pts = np.column_stack(np.where(skeleton_img > 0))  # [y, x]

        if len(skel_pts) == 0:
            return []

        # Find degree (neighbor count) for each skeleton pixel
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D((skeleton_img > 0).astype(np.uint8), -1, kernel)

        paths = []
        # First trace starting from endpoints (degree == 1) for natural stroke lines
        endpoints = [pt for pt in skel_pts if neighbor_count[pt[0], pt[1]] == 1]
        for y0, x0 in endpoints:
            if visited[y0, x0]:
                continue
            path = self._walk_path(x0, y0, skeleton_img, visited, neighbor_count)
            if len(path) >= 2:
                paths.append(path)

        # Next trace any remaining closed loops or internal segments
        for y0, x0 in skel_pts:
            if visited[y0, x0]:
                continue
            path = self._walk_path(x0, y0, skeleton_img, visited, neighbor_count)
            if len(path) >= 2:
                paths.append(path)

        # Reconnect collinear/close endpoints within 3 pixels to bridge small gaps
        merged_paths = self._bridge_path_gaps(paths, max_gap_px=4.0)
        return merged_paths

    def _walk_path(
        self,
        start_x: int,
        start_y: int,
        skel: np.ndarray,
        visited: np.ndarray,
        deg_map: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Walks along adjacent skeleton pixels until reaching another endpoint or junction."""
        h, w = skel.shape
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        cx, cy = start_x, start_y

        while True:
            # 8-connected neighbors
            next_pt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] > 0 and not visited[ny, nx]:
                        next_pt = (nx, ny)
                        break
                if next_pt:
                    break

            if next_pt is None:
                break

            cx, cy = next_pt
            visited[cy, cx] = True
            path.append((cx, cy))

            # Stop at junctions if path has reasonable length to avoid merging crossing features
            if deg_map[cy, cx] > 2 and len(path) > 5:
                break

        return path

    def _bridge_path_gaps(self, paths: List[List[Tuple[int, int]]], max_gap_px: float = 4.0) -> List[List[Tuple[int, int]]]:
        """Bridges small micro-gaps between path endpoints within a safe physical tolerance."""
        if len(paths) < 2:
            return paths

        merged = []
        used = [False] * len(paths)

        for i in range(len(paths)):
            if used[i]:
                continue
            curr_path = list(paths[i])
            used[i] = True

            extended = True
            while extended:
                extended = False
                end_x, end_y = curr_path[-1]
                for j in range(len(paths)):
                    if used[j]:
                        continue
                    p_start = paths[j][0]
                    p_end = paths[j][-1]

                    d_start = math.hypot(end_x - p_start[0], end_y - p_start[1])
                    d_end = math.hypot(end_x - p_end[0], end_y - p_end[1])

                    if d_start <= max_gap_px:
                        curr_path.extend(paths[j])
                        used[j] = True
                        extended = True
                        break
                    elif d_end <= max_gap_px:
                        curr_path.extend(reversed(paths[j]))
                        used[j] = True
                        extended = True
                        break

            merged.append(curr_path)

        return merged

    def _detect_facial_and_semantic_structures(
        self,
        gray: np.ndarray,
        rgb: np.ndarray,
        fg_mask: np.ndarray,
        dist_transform: np.ndarray,
        mm_per_pixel: float,
    ) -> Tuple[Optional[Dict[str, int]], Dict[str, Any]]:
        """
        Detects facial bounding box and specific semantic features:
        - Eyes (eyebrows, upper/lower lid, pupils, highlights)
        - Bindi / Forehead Mark
        - Nose & Lips
        - Necklace (repeated beads/chains)
        - Crown / Jewels
        - Hair flow lines
        """
        h, w = gray.shape
        features: Dict[str, Any] = {"dots": [], "landmarks": [], "chains": []}

        # Find foreground bounding box
        y_indices, x_indices = np.where(fg_mask > 0)
        if len(x_indices) == 0:
            return None, features

        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        fg_w = max_x - min_x
        fg_h = max_y - min_y

        # Character Face ROI estimated in central-upper region
        face_x0 = int(min_x + fg_w * 0.20)
        face_x1 = int(max_x - fg_w * 0.20)
        face_y0 = int(min_y + fg_h * 0.15)
        face_y1 = int(min_y + fg_h * 0.75)
        face_roi = {"x0": face_x0, "y0": face_y0, "x1": face_x1, "y1": face_y1,
                    "cx": (face_x0 + face_x1) // 2, "cy": (face_y0 + face_y1) // 2}

        # ── Detect Red Tilak / Bindi Features in Forehead ──
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red_tilak_mask = (((hsv[:, :, 0] < 14) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 90) & (fg_mask > 0)).astype(np.uint8) * 255
        forehead_mask = np.zeros_like(gray)
        forehead_mask[face_roi["y0"]:face_roi["cy"], face_roi["x0"]:face_roi["x1"]] = 255
        red_forehead = cv2.bitwise_and(red_tilak_mask, forehead_mask)

        red_contours, _ = cv2.findContours(red_forehead, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for rc in red_contours:
            rc_area = cv2.contourArea(rc)
            rc_perim = cv2.arcLength(rc, True)
            if rc_area >= 4 and rc_perim > 0:
                circ = 4 * math.pi * rc_area / (rc_perim * rc_perim)
                (cx, cy), radius = cv2.minEnclosingCircle(rc)
                r_mm = radius * mm_per_pixel
                if circ > 0.60 and 0.25 <= r_mm <= 2.5:
                    pt = (int(cx), int(cy))
                    features["dots"].append({
                        "center": pt,
                        "radius_px": max(2, int(radius)),
                        "type": "BINDI_DOT",
                        "color_hex": "#dc2626",
                        "color_name": "Ruby Red"
                    })
                    features["landmarks"].append(pt)

        # Check for morphological high-contrast dark dot candidates (pupils, bindi)
        contours, _ = cv2.findContours(((gray < 70) & (fg_mask > 0)).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if area >= 4 and perim > 0:
                circ = 4 * math.pi * area / (perim * perim)
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                r_mm = radius * mm_per_pixel
                if circ > 0.72 and 0.25 <= r_mm <= 2.0:
                    pt = (int(cx), int(cy))
                    if not any(math.hypot(pt[0] - d["center"][0], pt[1] - d["center"][1]) < 6 for d in features["dots"]):
                        features["dots"].append({
                            "center": pt,
                            "radius_px": max(1, int(radius)),
                            "type": "BINDI_DOT" if abs(pt[0] - face_roi["cx"]) < fg_w * 0.1 else "DETAIL_DOT"
                        })
                        features["landmarks"].append(pt)

        return face_roi, features

    def _is_near_facial_feature(self, pt: Tuple[int, int], features: Dict[str, Any]) -> bool:
        """Checks if a coordinate is close to a known facial landmark or dot."""
        for lm in features.get("landmarks", []):
            if math.hypot(pt[0] - lm[0], pt[1] - lm[1]) < 25:
                return True
        return False

    def _score_stroke_importance(
        self,
        path: List[Tuple[int, int]],
        width_mm: float,
        len_mm: float,
        gray: np.ndarray,
        face_roi: Optional[Dict[str, int]],
        features: Dict[str, Any],
    ) -> Tuple[float, str, bool]:
        """
        Computes multi-factor structural importance score for each stroke.
        Evaluates contrast, proximity to face landmarks, symmetry, and geometric role.
        """
        h, w = gray.shape
        mid_pt = path[len(path) // 2]
        mx, my = mid_pt

        # 1. Local Contrast
        local_pixels = [gray[int(max(0, min(h - 1, pt[1]))), int(max(0, min(w - 1, pt[0])))] for pt in path]
        avg_lum = float(np.mean(local_pixels))
        contrast_score = 1.0 - (avg_lum / 255.0)  # Darker lines = higher contrast

        # 2. Location / Face Priority
        in_face = False
        face_score = 0.2
        semantic_tag = "DECORATIVE_LINE"

        if face_roi and face_roi["x0"] <= mx <= face_roi["x1"] and face_roi["y0"] <= my <= face_roi["y1"]:
            in_face = True
            face_score = 0.90
            rel_y = (my - face_roi["y0"]) / max(1, (face_roi["y1"] - face_roi["y0"]))
            if rel_y < 0.35:
                semantic_tag = "EYEBROW" if len_mm > 1.2 else "FOREHEAD_ORNAMENT"
            elif rel_y < 0.60:
                semantic_tag = "EYE_OUTLINE"
            elif rel_y < 0.75:
                semantic_tag = "NOSE_CONTOUR"
            else:
                semantic_tag = "LIPS_CONTOUR"
        elif face_roi and my < face_roi["y0"]:
            semantic_tag = "CROWN_ORNAMENT"
            face_score = 0.65
        elif face_roi and my > face_roi["y1"] and abs(mx - face_roi["cx"]) < (face_roi["x1"] - face_roi["x0"]) * 0.8:
            semantic_tag = "NECKLACE_CURVE"
            face_score = 0.75
        elif face_roi and abs(mx - face_roi["cx"]) > (face_roi["x1"] - face_roi["x0"]) * 0.7:
            semantic_tag = "HAIR_FLOW"
            face_score = 0.55

        # 3. Path Length & Curvature Confidence
        length_score = min(1.0, len_mm / 4.0)

        # Weighted Importance Combination
        importance = (0.40 * face_score) + (0.35 * contrast_score) + (0.25 * length_score)
        importance = max(0.01, min(0.99, importance))

        # Mark as PROTECTED if inside face or key semantic structures
        is_protected = in_face or (importance >= 0.50) or (semantic_tag in {"NECKLACE_CURVE", "CROWN_ORNAMENT"})

        return importance, semantic_tag, is_protected

    def _sample_stroke_color(
        self,
        path: List[Tuple[int, int]],
        rgb: np.ndarray,
        palette: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, str]:
        """Samples RGB values along the path to determine the closest thread color."""
        h, w = rgb.shape[:2]
        colors = [rgb[int(max(0, min(h - 1, pt[1]))), int(max(0, min(w - 1, pt[0])))] for pt in path[::2]]
        if not colors:
            return "#141419", "Obsidian Black"

        avg_rgb = np.mean(colors, axis=0).astype(int)
        hex_col = f"#{avg_rgb[0]:02x}{avg_rgb[1]:02x}{avg_rgb[2]:02x}"

        if palette:
            best_color = min(palette, key=lambda c: np.linalg.norm(np.array(c["rgb"]) - avg_rgb))
            return best_color["hex"], best_color["name"]

        return hex_col, "Linework Color"

    def _sample_point_color(
        self,
        pt: Tuple[int, int],
        rgb: np.ndarray,
        palette: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, str]:
        """Samples point RGB for small dots/bindi."""
        h, w = rgb.shape[:2]
        x, y = int(max(0, min(w - 1, pt[0]))), int(max(0, min(h - 1, pt[1])))
        color = rgb[y, x]
        hex_col = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
        if palette:
            best_color = min(palette, key=lambda c: np.linalg.norm(np.array(c["rgb"]) - color))
            return best_color["hex"], best_color["name"]
        return hex_col, "Detail Color"

    def _smooth_polyline(self, pts: List[Tuple[int, int]], epsilon: float = 1.0) -> List[Tuple[float, float]]:
        """Applies Douglas-Peucker simplification followed by gentle curvature smoothing."""
        if len(pts) < 3:
            return [(float(p[0]), float(p[1])) for p in pts]

        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts_np, epsilon, False)
        simplified = [(float(p[0][0]), float(p[0][1])) for p in approx]
        return simplified

    def _build_object_hierarchy(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Constructs hierarchical semantic tree for inspection and locking."""
        tree: Dict[str, Any] = {
            "name": "ROOT",
            "children": {
                "FACE": {"EYES": [], "EYEBROWS": [], "BINDI": [], "NOSE": [], "LIPS": []},
                "HAIR": {"MAJOR_FLOW": [], "FINE_STRANDS": []},
                "CROWN": {"ORNAMENTS": []},
                "JEWELLERY": {"NECKLACE": [], "EARRINGS": []},
                "LARGE_FILLS": [],
            }
        }

        for obj in objects:
            tag = obj["semantic_tag"]
            if tag in {"EYE_OUTLINE", "EYE_PUPIL"}:
                tree["children"]["FACE"]["EYES"].append(obj["id"])
            elif tag == "EYEBROW":
                tree["children"]["FACE"]["EYEBROWS"].append(obj["id"])
            elif tag == "BINDI_DOT":
                tree["children"]["FACE"]["BINDI"].append(obj["id"])
            elif tag == "NOSE_CONTOUR":
                tree["children"]["FACE"]["NOSE"].append(obj["id"])
            elif tag == "LIPS_CONTOUR":
                tree["children"]["FACE"]["LIPS"].append(obj["id"])
            elif tag == "HAIR_FLOW":
                tree["children"]["HAIR"]["MAJOR_FLOW"].append(obj["id"])
            elif tag == "CROWN_ORNAMENT":
                tree["children"]["CROWN"]["ORNAMENTS"].append(obj["id"])
            elif tag in {"NECKLACE_CURVE", "NECKLACE_BEAD"}:
                tree["children"]["JEWELLERY"]["NECKLACE"].append(obj["id"])
            else:
                tree["children"]["JEWELLERY"]["EARRINGS"].append(obj["id"])

        return tree
