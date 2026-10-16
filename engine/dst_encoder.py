"""
POWER DIGITIZE — DST Encoder
Generates real Tajima DST embroidery files using pyembroidery.
"""

import json
import os
from pathlib import Path
from typing import List
import pyembroidery
from engine.stitch_generators import Stitch, StitchBlock

class DSTEncoder:
    """
    Encode stitch data into DST binary format using pyembroidery.
    Fixes earlier bit-packing corruption issues and properly groups color changes.
    """

    @classmethod
    def encode(cls, stitch_blocks: List[StitchBlock], output_path: str):
        """
        Encode stitch blocks into a DST file.
        
        Args:
            stitch_blocks: List of StitchBlock objects with stitch coordinates
            output_path: Path to write the .dst file
        """
        if not stitch_blocks:
            raise ValueError("No stitches to encode")

        pattern = pyembroidery.EmbPattern()
        current_color_hex = None
        previous_position = None
        segment_colors = []

        for block in stitch_blocks:
            if not block.stitches:
                continue

            # Handle Color Changes
            if current_color_hex is None:
                current_color_hex = block.color
                segment_colors.append(block.color.lower())
                cls._add_thread(pattern, block.color)
            elif current_color_hex != block.color:
                # Issue TRIM then COLOR_CHANGE
                position = previous_position or (0.0, 0.0)
                pattern.add_stitch_absolute(pyembroidery.TRIM, *position)
                pattern.add_stitch_absolute(pyembroidery.COLOR_CHANGE, *position)
                current_color_hex = block.color
                segment_colors.append(block.color.lower())
                cls._add_thread(pattern, block.color)

            first = block.stitches[0]
            first_position = (first.x * 10.0, first.y * 10.0)
            if (
                previous_position is not None
                and not (first.flags & Stitch.JUMP)
                and first_position != previous_position
            ):
                dist_mm = ((first_position[0] - previous_position[0]) ** 2 + (first_position[1] - previous_position[1]) ** 2) ** 0.5 / 10.0
                # Issue machine TRIM for moves between blocks > 3.0mm
                if dist_mm > 3.0:
                    pattern.add_stitch_absolute(pyembroidery.TRIM, *previous_position)
                pattern.add_stitch_absolute(pyembroidery.JUMP, *first_position)

            # Handle Stitches in this block
            for stitch in block.stitches:
                # pyembroidery expects coordinates in 0.1mm units.
                x_units = stitch.x * 10.0
                y_units = stitch.y * 10.0

                cmd = pyembroidery.STITCH
                if stitch.flags & Stitch.TRIM:
                    cmd = pyembroidery.TRIM
                elif stitch.flags & Stitch.JUMP:
                    if previous_position is not None:
                        jump_d_mm = (((x_units - previous_position[0]) ** 2 + (y_units - previous_position[1]) ** 2) ** 0.5) / 10.0
                        if jump_d_mm > 3.0:
                            pattern.add_stitch_absolute(pyembroidery.TRIM, *previous_position)
                    cmd = pyembroidery.JUMP
                    
                pattern.add_stitch_absolute(cmd, x_units, y_units)
                previous_position = (x_units, y_units)

        # Write to DST format
        pyembroidery.write_dst(pattern, output_path)
        metadata_path = Path(output_path).with_suffix(".colors.json")
        temporary_path = metadata_path.with_suffix(".colors.json.tmp")
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            json.dump({"segments": segment_colors}, metadata_file)
        os.replace(temporary_path, metadata_path)

    @classmethod
    def _add_thread(cls, pattern: pyembroidery.EmbPattern, color_hex: str):
        """Parse a hex color string and add it to the pattern's threadlist."""
        thread = pyembroidery.EmbThread()
        hex_str = color_hex.lstrip('#')
        if len(hex_str) == 6:
            # Set thread color (R, G, B)
            thread.color = (
                int(hex_str[0:2], 16),
                int(hex_str[2:4], 16),
                int(hex_str[4:6], 16)
            )
        pattern.add_thread(thread)
