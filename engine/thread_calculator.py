"""
POWER DIGITIZE — Thread Calculator
Calculates thread usage and cost estimates.
"""

import math
from typing import Dict, Any
import pyembroidery


class ThreadCalculator:
    """Calculate thread usage and cost for embroidery designs."""

    def calculate(
        self,
        dst_path: str,
        cost_per_meter: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Calculate thread usage per color and total cost.
        
        Args:
            dst_path: Path to DST file
            cost_per_meter: Cost per meter of thread (INR)
        """
        pattern = pyembroidery.read(dst_path)
        if pattern is None:
            raise ValueError("Cannot read DST file")

        # Track thread per color
        color_threads = {}
        current_color = 0
        prev_x, prev_y = 0.0, 0.0
        color_length = 0.0
        previous_was_stitch = False

        for x, y, cmd in pattern.stitches:
            x_mm = x / 10.0
            y_mm = y / 10.0
            cmd_type = cmd & 0xFF

            if cmd_type == pyembroidery.STITCH:
                if previous_was_stitch:
                    dx = x_mm - prev_x
                    dy = y_mm - prev_y
                    color_length += math.sqrt(dx*dx + dy*dy)
                previous_was_stitch = True
            elif cmd_type == pyembroidery.COLOR_CHANGE:
                # Save current color thread
                color_name = f"Color {current_color + 1}"
                if current_color < len(pattern.threadlist):
                    thread = pattern.threadlist[current_color]
                    if thread and thread.description:
                        color_name = thread.description

                color_threads[color_name] = {
                    "length_mm": color_length,
                    "length_m": round(color_length / 1000, 1),
                    "cost": round(color_length / 1000 * cost_per_meter, 2),
                }
                current_color += 1
                color_length = 0.0
                previous_was_stitch = False
            elif cmd_type in (pyembroidery.JUMP, pyembroidery.TRIM):
                previous_was_stitch = False

            prev_x, prev_y = x_mm, y_mm

        # Add last color if it has length
        if color_length > 0:
            color_name = f"Color {current_color + 1}"
            if current_color < len(pattern.threadlist):
                thread = pattern.threadlist[current_color]
                if thread and thread.description:
                    color_name = thread.description

            color_threads[color_name] = {
                "length_mm": color_length,
                "length_m": round(color_length / 1000, 1),
                "cost": round(color_length / 1000 * cost_per_meter, 2),
            }

        # Totals
        total_length_mm = sum(c["length_mm"] for c in color_threads.values())
        total_length_m = round(total_length_mm / 1000, 1)
        total_cost = round(total_length_m * cost_per_meter, 2)

        # Add bobbin thread estimate (roughly 60% of top thread for fills)
        bobbin_estimate_m = round(total_length_m * 0.6, 1)
        bobbin_cost = round(bobbin_estimate_m * cost_per_meter * 0.5, 2)  # Bobbin is cheaper

        return {
            "colors": color_threads,
            "total_thread_m": total_length_m,
            "total_cost": total_cost,
            "bobbin_thread_m": bobbin_estimate_m,
            "bobbin_cost": bobbin_cost,
            "grand_total_cost": round(total_cost + bobbin_cost, 2),
            "cost_per_meter": cost_per_meter,
            "currency": "INR",
        }
