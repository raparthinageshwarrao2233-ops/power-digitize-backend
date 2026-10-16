"""
POWER DIGITIZE — DST Validator & AI Embroidery Doctor
Analyzes and validates DST files for quality issues.
"""

import math
from typing import Dict, Any, List
import pyembroidery


class DSTValidator:
    """Validate and analyze DST embroidery files."""

    def analyze_dst(self, dst_path: str) -> Dict[str, Any]:
        """Read and analyze a DST file, return statistics."""
        pattern = pyembroidery.read(dst_path)

        if pattern is None:
            raise ValueError("Cannot read DST file")

        stitches = pattern.stitches
        stitch_count = 0
        jump_count = 0
        trim_count = 0
        color_changes = 0

        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        thread_length = 0.0
        prev_x, prev_y = 0.0, 0.0
        previous_was_stitch = False
        stitch_lengths = []

        for i, (x, y, cmd) in enumerate(stitches):
            # Convert from pyembroidery units (0.1mm) to mm
            x_mm = x / 10.0
            y_mm = y / 10.0

            min_x = min(min_x, x_mm)
            min_y = min(min_y, y_mm)
            max_x = max(max_x, x_mm)
            max_y = max(max_y, y_mm)

            cmd_type = cmd & 0xFF

            if cmd_type == pyembroidery.STITCH:
                stitch_count += 1
                if previous_was_stitch:
                    dx = x_mm - prev_x
                    dy = y_mm - prev_y
                    length = math.sqrt(dx*dx + dy*dy)
                    thread_length += length
                    stitch_lengths.append(length)
                previous_was_stitch = True
            elif cmd_type == pyembroidery.JUMP:
                jump_count += 1
                previous_was_stitch = False
            elif cmd_type == pyembroidery.TRIM:
                trim_count += 1
                previous_was_stitch = False
            elif cmd_type == pyembroidery.COLOR_CHANGE:
                color_changes += 1
                previous_was_stitch = False

            prev_x, prev_y = x_mm, y_mm

        width = max_x - min_x if min_x != float('inf') else 0
        height = max_y - min_y if min_y != float('inf') else 0

        # Average stitch length
        avg_stitch = sum(stitch_lengths) / max(1, len(stitch_lengths))
        min_stitch = min(stitch_lengths) if stitch_lengths else 0
        max_stitch = max(stitch_lengths) if stitch_lengths else 0

        # Estimated time (800 stitches/min)
        est_time = max(1, round(stitch_count / 800))

        running_count = 0
        satin_count = 0
        fill_count = 0
        underlay_count = 0
        repeat_count = 0

        # Read colors metadata if available
        color_segments = []
        try:
            from pathlib import Path
            import json
            meta_path = Path(dst_path).with_suffix(".colors.json")
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as mf:
                    color_segments = json.load(mf).get("segments", [])
        except Exception:
            pass

        return {
            "stitch_count": stitch_count,
            "jump_count": jump_count,
            "trim_count": trim_count,
            "color_count": color_changes + 1,
            "color_segments": color_segments,
            "width": round(width, 1),
            "height": round(height, 1),
            "min_x": round(min_x, 2) if min_x != float('inf') else 0.0,
            "max_x": round(max_x, 2) if max_x != float('-inf') else 0.0,
            "min_y": round(min_y, 2) if min_y != float('inf') else 0.0,
            "max_y": round(max_y, 2) if max_y != float('-inf') else 0.0,
            "center_x": round((min_x + max_x) / 2.0, 2) if min_x != float('inf') else 0.0,
            "center_y": round((min_y + max_y) / 2.0, 2) if min_y != float('inf') else 0.0,
            "thread_length": round(thread_length / 1000, 1),  # meters
            "estimated_time": est_time,
            "avg_stitch_length": round(avg_stitch, 2),
            "min_stitch_length": round(min_stitch, 2),
            "max_stitch_length": round(max_stitch, 2),
            "total_records": len(stitches),
        }

    def validate_roundtrip(
        self, dst_path: str, requested_width_mm: float, requested_height_mm: float
    ) -> Dict[str, Any]:
        """Verify DST round-trip and physical dimensions."""
        stats = self.analyze_dst(dst_path)
        gen_w = stats["width"]
        gen_h = stats["height"]

        error_w = round(abs(gen_w - requested_width_mm), 2)
        error_h = round(abs(gen_h - requested_height_mm), 2)
        tolerance_mm = 1.5

        # Valid if: non-zero stitches, fits within requested bounds (+ tolerance),
        # and at least one dimension matches the requested bounding limit (uniform aspect fit)
        fits_bounds = (gen_w <= requested_width_mm + tolerance_mm and gen_h <= requested_height_mm + tolerance_mm)
        dimension_anchored = (error_w <= tolerance_mm or error_h <= tolerance_mm)
        is_valid = bool(stats["stitch_count"] > 0 and fits_bounds and dimension_anchored)

        return {
            "roundtrip_valid": is_valid,
            "requested_size_mm": {"width": round(requested_width_mm, 1), "height": round(requested_height_mm, 1)},
            "generated_bounds_mm": {"width": round(gen_w, 1), "height": round(gen_h, 1)},
            "size_error_mm": {"width_error": error_w, "height_error": error_h},
            "tolerance_mm": tolerance_mm,
            "stitch_count": stats["stitch_count"],
            "jump_count": stats["jump_count"],
            "trim_count": stats["trim_count"],
            "status": "PASSED" if is_valid else "WARNING",
        }

    def validate(self, dst_path: str) -> Dict[str, Any]:
        """
        Validate a DST file and return issues.
        AI Embroidery Doctor — checks for common problems.
        """
        stats = self.analyze_dst(dst_path)
        issues: List[Dict[str, str]] = []

        if 0 < stats["stitch_count"] < 100:
            issues.append({
                "type": "warning",
                "icon": "⚠",
                "message": f"Only {stats['stitch_count']} stitches detected; confirm the design is large enough for the selected hoop.",
                "fixable": True,
            })

        # ── Check 1: Tiny stitches ──
        if stats["min_stitch_length"] < 0.5 and stats["stitch_count"] > 50:
            tiny_count = self._count_tiny_stitches(dst_path, 0.5)
            if tiny_count > 10:
                issues.append({
                    "type": "warning",
                    "icon": "⚠",
                    "message": f"Potentially excessive small stitches ({tiny_count} stitches < 0.5mm).",
                    "fixable": True,
                })

        # ── Check 2: Long jumps ──
        long_jumps = self._count_long_jumps(dst_path, 20.0)
        if long_jumps > 5:
            issues.append({
                "type": "warning",
                "icon": "⚠",
                "message": f"{long_jumps} long jump movements detected (> 20mm).",
                "fixable": True,
            })

        # ── Check 3: Excessive jumps ──
        if stats["jump_count"] > stats["stitch_count"] * 0.05:
            issues.append({
                "type": "warning",
                "icon": "⚠",
                "message": f"{stats['jump_count']} jump movements detected; review travel paths and trim placement.",
                "fixable": True,
            })

        # ── Check 4: High stitch density ──
        area_sq_mm = stats["width"] * stats["height"]
        if area_sq_mm > 0:
            density = stats["stitch_count"] / area_sq_mm
            if density > 5.0:
                issues.append({
                    "type": "warning",
                    "icon": "⚠",
                    "message": "High stitch density detected. May cause puckering or thread breaks.",
                    "fixable": True,
                })

        # ── Check 5: Hoop compatibility ──
        common_hoops = [
            ("50×50 mm", 50, 50),
            ("100×100 mm", 100, 100),
            ("130×180 mm", 130, 180),
            ("200×300 mm", 200, 300),
        ]
        compatible_hoops = []
        for name, hw, hh in common_hoops:
            if stats["width"] <= hw and stats["height"] <= hh:
                compatible_hoops.append(name)

        if not compatible_hoops:
            issues.append({
                "type": "problem",
                "icon": "✕",
                "message": f"Design ({stats['width']}×{stats['height']} mm) exceeds all standard hoops.",
                "fixable": False,
            })
        else:
            largest_incompatible = [
                (name, hw, hh) for name, hw, hh in common_hoops
                if stats["width"] > hw or stats["height"] > hh
            ]
            # Only warn if the design is incompatible with LARGE hoops (>=100mm), not just the tiny 50mm one
            big_incompatible = [n for n, hw, hh in largest_incompatible if hw >= 100]
            if big_incompatible:
                issues.append({
                    "type": "warning",
                    "icon": "⚠",
                    "message": f"Design exceeds {big_incompatible[-1]} hoop.",
                    "fixable": False,
                })

        # ── Check 6: Excessive color changes ──
        if stats["color_count"] > 15:
            issues.append({
                "type": "warning",
                "icon": "⚠",
                "message": f"Excessive color changes ({stats['color_count']}). May slow production.",
                "fixable": False,
            })

        # ── Good signs ──
        if stats["stitch_count"] > 0 and len(issues) == 0:
            issues.append({
                "type": "good",
                "icon": "✓",
                "message": "Design passes all quality checks.",
                "fixable": False,
            })

        if stats["avg_stitch_length"] >= 1.5 and stats["avg_stitch_length"] <= 5.0:
            issues.append({
                "type": "good",
                "icon": "✓",
                "message": f"Average stitch length ({stats['avg_stitch_length']}mm) is within optimal range.",
                "fixable": False,
            })

        # Sort: problems first, then warnings, then good
        order = {"problem": 0, "warning": 1, "good": 2}
        issues.sort(key=lambda x: order.get(x["type"], 1))

        return {
            "stats": stats,
            "issues": issues,
            "compatible_hoops": compatible_hoops,
            "overall": "problem" if any(i["type"] == "problem" for i in issues)
                       else "warning" if any(i["type"] == "warning" for i in issues)
                       else "good",
        }

    def _count_tiny_stitches(self, dst_path: str, threshold: float) -> int:
        """Count stitches shorter than threshold."""
        pattern = pyembroidery.read(dst_path)
        count = 0
        prev_x, prev_y = 0, 0
        previous_was_stitch = False

        for x, y, cmd in pattern.stitches:
            x_mm, y_mm = x / 10.0, y / 10.0
            command = cmd & 0xFF
            if command == pyembroidery.STITCH:
                if not previous_was_stitch:
                    prev_x, prev_y = x_mm, y_mm
                    previous_was_stitch = True
                    continue
                dx = x_mm - prev_x
                dy = y_mm - prev_y
                if math.sqrt(dx*dx + dy*dy) < threshold:
                    count += 1
                previous_was_stitch = True
            else:
                previous_was_stitch = False
            prev_x, prev_y = x_mm, y_mm

        return count

    def _count_long_jumps(self, dst_path: str, threshold: float) -> int:
        """Count jumps longer than threshold."""
        pattern = pyembroidery.read(dst_path)
        count = 0
        prev_x, prev_y = 0, 0

        for x, y, cmd in pattern.stitches:
            x_mm, y_mm = x / 10.0, y / 10.0
            if (cmd & 0xFF) == pyembroidery.JUMP:
                dx = x_mm - prev_x
                dy = y_mm - prev_y
                if math.sqrt(dx*dx + dy*dy) > threshold:
                    count += 1
            prev_x, prev_y = x_mm, y_mm

        return count

    def validate_stitch_blocks(
        self,
        stitch_blocks,
        min_stitch_mm: float = 0.3,
        max_stitch_mm: float = 12.7,
        max_satin_width_mm: float = 12.0,
    ) -> Dict[str, Any]:
        """
        Validate stitch blocks before DST export.
        Checks: duplicate paths, stitch length limits, jump density.
        Returns validation report dict.
        """
        issues: list = []
        total_stitches = 0
        tiny_count = 0
        long_count = 0
        duplicate_count = 0

        # Duplicate detection via path signatures
        sigs = []
        for block in stitch_blocks:
            normals = [(s.x, s.y) for s in block.stitches if s.flags == 0]
            total_stitches += len(normals)

            # Tiny / long stitch checks
            for i in range(1, len(normals)):
                dx = normals[i][0] - normals[i-1][0]
                dy = normals[i][1] - normals[i-1][1]
                length = math.sqrt(dx*dx + dy*dy)
                if length < min_stitch_mm and length > 0.01:
                    tiny_count += 1
                if length > max_stitch_mm:
                    long_count += 1

            # Duplicate path signature
            if block.stitch_type not in ("underlay", "repeat") and len(normals) >= 3:
                mid = normals[len(normals) // 2]
                sig = (
                    round(normals[0][0], 1), round(normals[0][1], 1),
                    round(normals[-1][0], 1), round(normals[-1][1], 1),
                    len(normals),
                    round(mid[0], 1), round(mid[1], 1),
                )
                for es in sigs:
                    if (abs(sig[0]-es[0]) < 0.5 and abs(sig[1]-es[1]) < 0.5
                        and abs(sig[2]-es[2]) < 0.5 and abs(sig[3]-es[3]) < 0.5
                        and abs(sig[4]-es[4]) < 3):
                        duplicate_count += 1
                        break
                sigs.append(sig)

        if tiny_count > total_stitches * 0.05:
            issues.append({
                "type": "warning", "icon": "⚠",
                "message": f"{tiny_count} stitches shorter than {min_stitch_mm}mm",
                "fixable": True,
            })
        if long_count > 0:
            issues.append({
                "type": "warning", "icon": "⚠",
                "message": f"{long_count} stitches exceed {max_stitch_mm}mm maximum",
                "fixable": True,
            })
        if duplicate_count > 0:
            issues.append({
                "type": "warning", "icon": "⚠",
                "message": f"{duplicate_count} duplicate stitch path(s) detected",
                "fixable": True,
            })

        passed = len(issues) == 0
        if passed:
            issues.append({
                "type": "good", "icon": "✓",
                "message": "All pre-export stitch checks passed.",
                "fixable": False,
            })

        return {
            "passed": passed,
            "total_stitches": total_stitches,
            "tiny_stitches": tiny_count,
            "long_stitches": long_count,
            "duplicate_paths": duplicate_count,
            "issues": issues,
        }

    def validate_pre_export(
        self,
        stitch_blocks,
        fidelity_result: dict = None,
    ) -> Dict[str, Any]:
        """
        Combined pre-export validation gate.
        Checks stitch quality + fidelity gates.
        Returns {passed: bool, failures: [...], ...}
        """
        stitch_val = self.validate_stitch_blocks(stitch_blocks)
        failures = []

        if not stitch_val["passed"]:
            for iss in stitch_val["issues"]:
                if iss["type"] in ("warning", "problem"):
                    failures.append(iss["message"])

        if fidelity_result:
            if not fidelity_result.get("passes_gate", True):
                for gf in fidelity_result.get("gate_failures", []):
                    failures.append(f"Fidelity: {gf}")

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "stitch_validation": stitch_val,
            "fidelity_gate": fidelity_result.get("passes_gate", True) if fidelity_result else None,
        }

