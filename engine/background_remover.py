"""
POWER DIGITIZE — Background Remover
AI-powered background removal using rembg (U²-Net deep learning model).
Falls back to enhanced OpenCV method if rembg is not available.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

# Try to import rembg for AI-powered background removal
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


class BackgroundRemover:
    """Remove background from artwork images for clean embroidery input."""

    def remove(self, image_path: str, output_path: str) -> dict:
        """
        Remove background from the image.
        Uses rembg (AI) if available, falls back to enhanced OpenCV.
        Returns info about the removal process.
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # If already has alpha, check if background is transparent
        has_alpha = len(img.shape) == 3 and img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            transparent_pct = np.sum(alpha < 128) / alpha.size * 100
            if transparent_pct > 10:
                # Already has usable transparency — refine edges and save
                refined = self._refine_alpha_edges(img)
                cv2.imwrite(output_path, refined)
                return {
                    "method": "existing_alpha_refined",
                    "removed_pct": round(transparent_pct, 1),
                }

        bgr = img[:, :, :3] if len(img.shape) == 3 and img.shape[2] >= 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        # Sample corners to detect background color
        corner_size = max(5, min(h, w) // 20)
        corners = [
            bgr[:corner_size, :corner_size],
            bgr[:corner_size, -corner_size:],
            bgr[-corner_size:, :corner_size],
            bgr[-corner_size:, -corner_size:],
        ]
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        corner_std = np.std(corner_pixels, axis=0).mean()

        # If solid or near-solid background (standard for logos, graphics, clip-art),
        # use enhanced solid background removal which preserves all interior holes and colors
        if corner_std < 35:
            result, method = self._remove_solid_bg(bgr)
            cv2.imwrite(output_path, result)
            removed_pct = np.sum(result[:, :, 3] < 128) / (result.shape[0] * result.shape[1]) * 100
            return {
                "method": method,
                "removed_pct": round(removed_pct, 1),
                "output_path": output_path,
            }

        # Strategy 1: AI-powered removal with rembg for complex/photographic backgrounds
        if REMBG_AVAILABLE:
            try:
                result, method = self._remove_ai(image_path)
                cv2.imwrite(output_path, result)
                removed_pct = np.sum(result[:, :, 3] < 128) / (result.shape[0] * result.shape[1]) * 100
                return {
                    "method": method,
                    "removed_pct": round(removed_pct, 1),
                    "output_path": output_path,
                }
            except Exception:
                pass  # Fall through to OpenCV method

        # Fallback OpenCV method
        result, method = self._remove_solid_bg(bgr)
        cv2.imwrite(output_path, result)
        removed_pct = np.sum(result[:, :, 3] < 128) / (result.shape[0] * result.shape[1]) * 100

        return {
            "method": method,
            "removed_pct": round(removed_pct, 1),
            "output_path": output_path,
        }

    def _remove_ai(self, image_path: str) -> Tuple[np.ndarray, str]:
        """AI-powered background removal using rembg (U-Net deep learning)."""
        from rembg import new_session
        
        # Load image with PIL (rembg expects PIL Image)
        pil_img = Image.open(image_path).convert("RGBA")

        # Create session with the smaller/faster u2netp model (~4MB)
        session = new_session("u2netp")

        # Use rembg to remove background without aggressive matting erosion
        result_pil = rembg_remove(
            pil_img,
            session=session,
            alpha_matting=False,
        )

        # Convert back to OpenCV format (BGRA)
        result_np = np.array(result_pil)
        # PIL returns RGBA, OpenCV uses BGRA
        result_bgra = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGRA)

        return result_bgra, "ai_rembg_u2netp"

    def _refine_alpha_edges(self, img: np.ndarray) -> np.ndarray:
        """Refine alpha channel edges while preserving fine anti-aliased tips."""
        alpha = img[:, :, 3].copy()

        # Clean noise: only cut away nearly transparent stray pixels (< 15)
        # and solidify semi-opaque pixels (> 180) to preserve delicate slender details like antlers/whiskers
        alpha[alpha > 180] = 255
        alpha[alpha < 15] = 0

        result = img.copy()
        result[:, :, 3] = alpha
        return result

    def _remove_solid_bg(self, bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        """Remove solid-colored background with enhanced edge detection."""
        h, w = bgr.shape[:2]

        # Sample corners to detect background color
        corner_size = max(5, min(h, w) // 20)
        corners = [
            bgr[:corner_size, :corner_size],              # top-left
            bgr[:corner_size, -corner_size:],              # top-right
            bgr[-corner_size:, :corner_size],              # bottom-left
            bgr[-corner_size:, -corner_size:],             # bottom-right
        ]

        # Average color of corners
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_color = np.median(corner_pixels, axis=0).astype(np.uint8)

        # Check if corners are consistent (solid background)
        corner_std = np.std(corner_pixels, axis=0).mean()

        if corner_std < 35:
            # Remove only background-colored pixels connected to the image edges.
            bg_color = np.median(corner_pixels, axis=0).astype(np.int16)
            distance = np.max(np.abs(bgr.astype(np.int16) - bg_color), axis=2)
            candidates = (distance <= 45).astype(np.uint8)
            _, labels = cv2.connectedComponents(candidates, connectivity=4)
            edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            edge_background = np.isin(labels, edge_labels[edge_labels != 0])
            color_background = distance <= 22
            bg_mask = (edge_background | color_background).astype(np.uint8) * 255

            # Preserve clean edges and interior negative spaces
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            fg_mask = cv2.bitwise_not(bg_mask)

            # Create RGBA output
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = fg_mask

            method = "enhanced_flood_fill"
        else:
            # Non-solid background — use edge-based approach with bilateral filter
            # Bilateral filter preserves edges while smoothing noise
            filtered = cv2.bilateralFilter(bgr, 9, 75, 75)
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Fill holes
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, contours, -1, 255, -1)

            # Smooth mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask

            method = "enhanced_threshold"

        return rgba, method
