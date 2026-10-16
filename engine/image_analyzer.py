"""
POWER DIGITIZE — Image Analyzer
Analyzes uploaded artwork for embroidery digitizing.
Uses edge-preserving bilateral filter, LAB color space quantization,
adaptive contour approximation, and PCA-based shape analysis.
"""

import math
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any


class ImageAnalyzer:
    """Analyzes an image to extract embroidery-relevant information."""

    def analyze(self, image_path: str, color_count: int = 8) -> Dict[str, Any]:
        """Full image analysis pipeline."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Convert BGRA/BGR to consistent format
        has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False

        if has_alpha:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ── EDGE-PRESERVING smoothing (Bilateral + Mean-Shift) ──
        # Eliminates gradient stepping and micro-holes while preserving crisp boundaries
        bgr_clean = bgr.copy()
        if alpha is not None:
            fg_m = (alpha > 20).astype(np.uint8) * 255
            bgr_clean = cv2.bitwise_and(bgr_clean, bgr_clean, mask=fg_m)

        smoothed_bgr = cv2.bilateralFilter(bgr_clean, d=7, sigmaColor=35, sigmaSpace=35)
        # Apply gentle pyramid mean-shift for continuous color regions without blurring fine details (<4px)
        smoothed_bgr = cv2.pyrMeanShiftFiltering(smoothed_bgr, sp=3, sr=14)
        if alpha is not None:
            smoothed_bgr = cv2.bitwise_and(smoothed_bgr, smoothed_bgr, mask=fg_m)
        smoothed_rgb = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB)

        # Extract global eye / interior detail candidates from artwork negative spaces
        global_eyes = []
        fg_mask = (alpha > 20).astype(np.uint8) * 255 if alpha is not None else (gray < 245).astype(np.uint8) * 255
        fg_cnts, fg_hier = cv2.findContours(fg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        if fg_hier is not None:
            for idx in range(len(fg_cnts)):
                if fg_hier[0][idx][3] >= 0:  # inner negative space / hole
                    c = fg_cnts[idx]
                    c_area = cv2.contourArea(c)
                    c_peri = cv2.arcLength(c, True)
                    c_circ = 4 * np.pi * c_area / max(c_peri**2, 1)
                    (c_cx, c_cy), c_r = cv2.minEnclosingCircle(c)
                    if 15 <= c_area <= 650 and c_circ >= 0.65 and 2.0 <= c_r <= 18.0:
                        global_eyes.append({
                            "center": [float(c_cx), float(c_cy)],
                            "radius_px": float(c_r),
                            "area_px": float(c_area),
                        })

        colors, labels = self._quantize_colors(smoothed_rgb, alpha, color_count)
        result = {
            "width": img.shape[1],
            "height": img.shape[0],
            "has_alpha": has_alpha,
            "colors": colors,
            "contours": self._extract_color_regions(labels, alpha, colors, smoothed_rgb),
            "complexity": self._assess_complexity(gray),
            "eye_candidates": global_eyes,
        }

        return result

    def _extract_clean_centerline(self, skeleton: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[List[int]]:
        """Extract a clean, non-reversing 1D centerline path from a 1px skeleton.
        
        Unlike cv2.findContours which traces both sides of a 1px line (doubling points
        and reversing direction), this walks the topological skeleton graph from one
        endpoint to the opposite end.
        """
        pts = np.argwhere(skeleton > 0)
        if len(pts) == 0:
            return []
        if len(pts) == 1:
            return [[int(pts[0][1] + offset[0]), int(pts[0][0] + offset[1])]]

        pt_set = set((int(r), int(c)) for r, c in pts)
        neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for r, c in pt_set:
            nbrs = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pt_set:
                        nbrs.append((nr, nc))
            neighbors[(r, c)] = nbrs

        # Endpoints have degree 1
        endpoints = [p for p, nbrs in neighbors.items() if len(nbrs) == 1]
        start = endpoints[0] if endpoints else min(pt_set)

        # Traverse graph greedily staying on main branch
        path = [start]
        visited = {start}
        curr = start
        while True:
            unvisited = [n for n in neighbors[curr] if n not in visited]
            if not unvisited:
                break
            # Pick neighbor with fewest unvisited neighbors (stay on main trail)
            unvisited.sort(key=lambda n: len([nn for nn in neighbors[n] if nn not in visited]))
            nxt = unvisited[0]
            visited.add(nxt)
            path.append(nxt)
            curr = nxt

        return [[int(c + offset[0]), int(r + offset[1])] for r, c in path]

    def _quantize_colors(
        self, rgb: np.ndarray, alpha: np.ndarray = None, color_count: int = 8
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Build a deterministic palette and assign every foreground pixel.
        
        Isolates solid core pixels (excluding edge anti-aliasing / bleed) to find
        genuine ink/thread colors. Matches colors via CIE Delta E 2000 and consolidates
        fringe artifacts into adjacent parent shapes.
        """
        from .thread_palette import match_color, _delta_e_2000

        height, width = rgb.shape[:2]
        foreground = alpha > 20 if alpha is not None else np.ones((height, width), dtype=bool)
        
        if alpha is not None:
            core = alpha > 180
            if np.count_nonzero(core) < 50:
                core = alpha > 60
            if np.count_nonzero(core) < 20:
                core = foreground
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            foreground = gray < 245
            grad = cv2.Laplacian(gray, cv2.CV_32F)
            core = foreground & (np.abs(grad) < 28)
            if np.count_nonzero(core) < 50:
                core = foreground

        core_pixels = rgb[core]
        if len(core_pixels) == 0:
            core_pixels = rgb[foreground]
        if len(core_pixels) == 0:
            empty = np.zeros((height, width), dtype=np.uint8)
            return [{"rgb": [0, 0, 0], "hex": "#000000", "percentage": 100.0, "name": "Black"}], empty

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        core_lab = lab[core] if np.count_nonzero(core) > 0 else lab[foreground]

        # Check if design is essentially single-color (low chromaticity variance)
        a_std = float(np.std(core_lab[:, 1]))
        b_std = float(np.std(core_lab[:, 2]))
        l_std = float(np.std(core_lab[:, 0]))

        # Stride sampling
        stride = max(1, len(core_lab) // 30000)
        sample = np.ascontiguousarray(core_lab[::stride], dtype=np.float32)
        unique_count = len(np.unique(sample.reshape(-1, 3), axis=0))
        
        if a_std < 6.0 and b_std < 6.0 and l_std < 22.0:
            # Clean single-color artwork (e.g. silhouette, monochrome emblem)
            k = 1
        else:
            k = max(1, min(int(color_count), unique_count, len(sample)))

        if k == 1:
            centers_lab = np.mean(sample, axis=0, keepdims=True, dtype=np.float32)
        else:
            cv2.setRNGSeed(42)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, _, centers_lab = cv2.kmeans(sample, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        # Convert centers from LAB to RGB
        palette_rgb = []
        for c_lab in centers_lab:
            c_img = np.uint8([[[c_lab[0], c_lab[1], c_lab[2]]]])
            c_bgr = cv2.cvtColor(c_img, cv2.COLOR_LAB2BGR)[0, 0]
            palette_rgb.append([int(c_bgr[2]), int(c_bgr[1]), int(c_bgr[0])])

        # Merge clusters that share chromatic hue or are within CIE Delta E 2000 threshold
        unique_centers_rgb: List[List[int]] = []
        unique_centers_lab: List[np.ndarray] = []
        cluster_remap: Dict[int, int] = {}

        for idx, rgb_val in enumerate(palette_rgb):
            matched_u_idx = None
            for u_idx, u_rgb in enumerate(unique_centers_rgb):
                de = _delta_e_2000(centers_lab[idx], unique_centers_lab[u_idx])
                da = abs(float(centers_lab[idx][1] - unique_centers_lab[u_idx][1]))
                db = abs(float(centers_lab[idx][2] - unique_centers_lab[u_idx][2]))
                chroma_dist = math.hypot(da, db)

                # Merge identical / shaded gradient colors of same hue family
                # For monochrome artwork (low chroma variance), merge close shades.
                # For multi-color vibrant artwork, keep distinct chromatic colors (de < 9.0)
                if a_std < 6.0 and b_std < 6.0:
                    should_merge = de < 16.0 or (chroma_dist < 8.0 and de < 28.0) or chroma_dist < 5.0
                else:
                    should_merge = de < 9.0 or (chroma_dist < 3.5 and de < 14.0)

                if should_merge:
                    matched_u_idx = u_idx
                    break

            if matched_u_idx is not None:
                cluster_remap[idx] = matched_u_idx
            else:
                new_idx = len(unique_centers_rgb)
                unique_centers_rgb.append(rgb_val)
                unique_centers_lab.append(centers_lab[idx])
                cluster_remap[idx] = new_idx

        k_final = len(unique_centers_rgb)
        final_centers_lab = np.array(unique_centers_lab, dtype=np.float32)

        # Assign all foreground pixels to the nearest consolidated color center
        flat_lab = lab.reshape(-1, 3)
        flat_foreground = foreground.reshape(-1)
        foreground_indices = np.flatnonzero(flat_foreground)

        labels = np.zeros((height, width), dtype=np.uint8)
        counts = np.zeros(k_final, dtype=np.int64)

        if k_final == 1:
            labels.reshape(-1)[foreground_indices] = 0
            counts[0] = len(foreground_indices)
        else:
            for start in range(0, len(foreground_indices), 50000):
                indices = foreground_indices[start:start + 50000]
                values = flat_lab[indices]
                distances = np.sum((values[:, None, :] - final_centers_lab[None, :, :]) ** 2, axis=2)
                chunk_labels = np.argmin(distances, axis=1).astype(np.uint8)
                labels.reshape(-1)[indices] = chunk_labels
                counts += np.bincount(chunk_labels, minlength=k_final)

            # Consolidate tiny boundary fringe clusters bordering dominant colors
            total_fg = max(1, len(foreground_indices))
            min_thresh = 4.5 if (a_std < 6.0 and b_std < 6.0) else 0.35
            for c_idx in range(k_final):
                pct = float(counts[c_idx]) / total_fg * 100.0
                if (pct < min_thresh or counts[c_idx] < 25) and k_final > 1:
                    # Find dominant color
                    dom_idx = int(np.argmax(counts))
                    if dom_idx != c_idx:
                        labels[labels == c_idx] = dom_idx
                        counts[dom_idx] += counts[c_idx]
                        counts[c_idx] = 0

        # Build final color metadata list
        total = max(1, int(counts.sum()))
        colors = []
        for index in range(k_final):
            if counts[index] == 0:
                continue
            color_rgb = unique_centers_rgb[index]
            red, green, blue = color_rgb[0], color_rgb[1], color_rgb[2]
            matched_thread = match_color((red, green, blue), brand="Madeira")

            colors.append({
                "id": index,
                "rgb": [red, green, blue],
                "hex": f"#{red:02x}{green:02x}{blue:02x}",
                "percentage": round(float(counts[index]) / total * 100, 1),
                "name": matched_thread.name if matched_thread else self._color_name(red, green, blue),
                "thread_code": matched_thread.code if matched_thread else f"RGB_{red}_{green}_{blue}",
                "thread_brand": matched_thread.brand if matched_thread else "Custom",
                "delta_e": matched_thread.delta_e if matched_thread else 0.0,
            })

        colors.sort(key=lambda color: color["percentage"], reverse=True)
        return colors, labels

    def _extract_color_regions(
        self, labels: np.ndarray, alpha: np.ndarray, colors: List[Dict[str, Any]],
        rgb: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        """Extract connected stitchable regions independently for each palette color.
        
        Uses CHAIN_APPROX_TC89_L1 for true curve geometry and fine adaptive epsilon
        so narrow gaps, horn tips, and detailed boundaries are preserved.
        """
        foreground = alpha > 20 if alpha is not None else np.ones(labels.shape, dtype=bool)
        min_area = max(12.0, labels.shape[0] * labels.shape[1] * 0.00001)
        shapes = []
        next_id = 0

        for color in colors:
            color_id = color["id"]
            mask = ((labels == color_id) & foreground).astype(np.uint8) * 255

            if len(colors) > 1:
                # 1px dilation ensures adjacent color regions overlap at physical seams, eliminating gaps
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                mask = cv2.dilate(mask, kernel, iterations=1) & (foreground.astype(np.uint8) * 255)

            # Clean tiny single-pixel noise without eroding genuine narrow tips
            num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask)
            for cc_idx in range(1, num_cc):
                if cc_stats[cc_idx, cv2.CC_STAT_AREA] < 6:
                    mask[cc_labels == cc_idx] = 0

            # Extract contours directly to preserve true negative spaces and narrow gaps
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            if hierarchy is None:
                continue

            for index, contour in enumerate(contours):
                if hierarchy[0][index][3] >= 0:
                    continue  # inner hole handled below
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                x, y, width, height = cv2.boundingRect(contour)
                max_dim = max(width, height, 1)

                # High-fidelity adaptive epsilon: preserves sharp antler tips and curves
                if area > 5000:
                    epsilon = max(0.4, 0.0010 * perimeter)
                elif area > 1000:
                    epsilon = max(0.4, 0.0015 * perimeter)
                elif area > 200:
                    epsilon = max(0.35, 0.0020 * perimeter)
                else:
                    epsilon = max(0.3, 0.0025 * perimeter)

                outer = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()

                holes = []
                eye_candidates = []
                hole_area = 0.0
                boundary_length = perimeter
                child = int(hierarchy[0][index][2])
                while child >= 0:
                    child_area = cv2.contourArea(contours[child])
                    if child_area >= min_area:
                        hole_area += child_area
                        boundary_length += cv2.arcLength(contours[child], True)
                        hole = cv2.approxPolyDP(contours[child], epsilon, True).reshape(-1, 2).tolist()
                        holes.append(hole)
                        # Check for eye candidate (strictly circular hole in upper/face region)
                        (cx_h, cy_h), r_h = cv2.minEnclosingCircle(contours[child])
                        circ_h = 4 * np.pi * child_area / max(cv2.arcLength(contours[child], True)**2, 1)
                        if 15 <= child_area <= 650 and circ_h >= 0.80 and 2.0 <= r_h <= 18.0:
                            eye_candidates.append({
                                "center": [float(cx_h), float(cy_h)],
                                "radius_px": float(r_h),
                                "area_px": float(child_area),
                            })
                    child = int(hierarchy[0][child][0])

                net_area = max(1.0, area - hole_area)
                effective_width = 2 * net_area / max(boundary_length, 1.0)
                thinness = min(1.0, effective_width / max_dim)
                circularity = 4 * np.pi * net_area / max(boundary_length * boundary_length, 1)
                shape_type = self._classify_shape(net_area, thinness, circularity, width, height, effective_width)

                # PCA-based optimal fill angle
                fill_angle = self._calculate_optimal_angle(contour)

                centerline = []
                if shape_type in {"thin_line", "narrow", "detail"}:
                    component = np.zeros((height + 4, width + 4), dtype=np.uint8)
                    shifted_outer = np.asarray(outer, dtype=np.int32) - np.array([x - 2, y - 2])
                    cv2.drawContours(component, [shifted_outer.reshape(-1, 1, 2)], -1, 255, -1)
                    for hole in holes:
                        shifted_hole = np.asarray(hole, dtype=np.int32) - np.array([x - 2, y - 2])
                        cv2.drawContours(component, [shifted_hole.reshape(-1, 1, 2)], -1, 0, -1)

                    skeleton = np.zeros_like(component)
                    work = component.copy()
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                    while cv2.countNonZero(work):
                        eroded = cv2.erode(work, kernel)
                        opened = cv2.dilate(eroded, kernel)
                        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
                        work = eroded

                    centerline = self._extract_clean_centerline(skeleton, offset=(x - 2, y - 2))

                shapes.append({
                    "id": next_id,
                    "color_id": color_id,
                    "bbox": {"x": int(x), "y": int(y), "w": int(width), "h": int(height)},
                    "area": float(net_area),
                    "perimeter": float(boundary_length),
                    "thinness": float(thinness),
                    "effective_width": float(effective_width),
                    "circularity": float(circularity),
                    "shape_type": shape_type,
                    "stitch_type": self._recommend_stitch_type(shape_type, area, thinness),
                    "fill_angle": float(fill_angle),
                    "holes": holes,
                    "eye_candidates": eye_candidates,
                    "centerline_points": centerline,
                    "contour_points": outer,
                })
                next_id += 1

        # Harmonize fill angles for lettering / wordmark lines sharing a horizontal baseline
        # (prevents disjointed optical sheen and mismatched pull on adjacent text characters)
        by_col = {}
        for s in shapes:
            by_col.setdefault(s["color_id"], []).append(s)

        for cid, slist in by_col.items():
            if len(slist) >= 3:
                h_tol = max(10, labels.shape[0] * 0.035)
                for s in slist:
                    base_y = s["bbox"]["y"] + s["bbox"]["h"]
                    cluster = [
                        other for other in slist
                        if abs((other["bbox"]["y"] + other["bbox"]["h"]) - base_y) <= h_tol
                        and other["area"] > 5000
                    ]
                    if len(cluster) >= 3:
                        for member in cluster:
                            member["fill_angle"] = 45.0

        shapes.sort(key=lambda shape: shape["area"], reverse=True)
        return shapes

    def _calculate_optimal_angle(self, contour: np.ndarray) -> float:
        """Calculate optimal fill angle using PCA (Principal Component Analysis).
        
        The fill direction is set perpendicular to the shape's principal axis,
        which ensures stitches follow the shape naturally.
        """
        try:
            pts = contour.reshape(-1, 2).astype(np.float64)
            if len(pts) < 3:
                return 0.0

            # Compute covariance matrix
            mean = np.mean(pts, axis=0)
            centered = pts - mean
            cov_matrix = np.cov(centered.T)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Principal axis = direction of maximum variance
            principal_idx = np.argmax(eigenvalues)
            principal_vec = eigenvectors[:, principal_idx]

            # Fill angle perpendicular to principal axis
            angle = np.degrees(np.arctan2(principal_vec[1], principal_vec[0]))
            fill_angle = (angle + 90) % 180  # Perpendicular, normalized to [0, 180)

            return fill_angle
        except Exception:
            return 0.0

    def _classify_shape(
        self, area: float, thinness: float, circularity: float, w: int, h: int, effective_width: float = 10.0
    ) -> str:
        """Classify a contour into a shape category."""
        if area < 30:
            return "detail"
        if (thinness < 0.10 and min(w, h) < 18) or effective_width < 4.0:
            return "thin_line"
        if (thinness < 0.28 and effective_width < 14.0) or (area < 800 and min(w, h) < 25):
            return "narrow"
        if circularity > 0.75 and area < 400:
            return "circular"
        return "region"

    def _recommend_stitch_type(
        self, shape_type: str, area: float, thinness: float
    ) -> str:
        """Recommend the best stitch type for a shape."""
        if shape_type == "thin_line":
            return "running"
        if shape_type == "narrow":
            return "satin"
        if shape_type in ("region", "circular"):
            return "fill"
        if shape_type == "detail":
            return "running" if area < 40 else "satin"
        return "fill"

    def _assess_complexity(self, gray: np.ndarray) -> Dict[str, Any]:
        """Assess overall image complexity for digitizing suitability."""
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Gradient complexity
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_complexity = np.mean(gradient_mag) / 255

        # Overall score (lower = simpler = better for embroidery)
        score = edge_density * 0.5 + gradient_complexity * 0.5

        if score < 0.05:
            rating = "excellent"
            message = "Clean artwork, ideal for automatic digitizing."
        elif score < 0.15:
            rating = "good"
            message = "Good artwork, suitable for automatic digitizing."
        elif score < 0.3:
            rating = "moderate"
            message = "Moderate complexity. Some manual adjustments may improve results."
        else:
            rating = "complex"
            message = "This artwork contains many gradients/details. Try simplifying the artwork for better results."

        return {
            "score": round(float(score), 4),
            "rating": rating,
            "message": message,
            "edge_density": round(float(edge_density), 4),
            "gradient_complexity": round(float(gradient_complexity), 4),
        }

    @staticmethod
    def _color_name(r: int, g: int, b: int) -> str:
        """Get a human-readable name for an RGB color."""
        # Simple color naming based on HSV
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        if v < 30:
            return "Black"
        if s < 30:
            if v > 200:
                return "White"
            if v > 120:
                return "Light Gray"
            return "Dark Gray"

        # Color ranges
        if h < 10 or h > 170:
            return "Red"
        if h < 25:
            return "Orange"
        if h < 35:
            return "Yellow"
        if h < 80:
            return "Green"
        if h < 100:
            return "Cyan"
        if h < 130:
            return "Blue"
        if h < 155:
            return "Purple"
        return "Pink"
