"""
Fabric Physics & Preset Engine.
Models physical embroidery distortion on real fabrics:
1. Pull Distortion: Thread tension shrinks fabric inward along the stitch angle.
2. Push Distortion: Thread packing pushes fabric outward perpendicular to the stitch angle.
3. Pile / Loft Sinking: High pile (fleece, terry) swallows stitches; requires dense tamping underlay.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FabricPreset:
    id: str
    label: str
    category: str
    description: str
    pull_compensation_mm: float
    push_compensation_mm: float
    recommended_density_mm: float
    recommended_stitch_length_mm: float
    min_stitch_length_mm: float
    recommended_underlay: str
    recommended_stabilizer: str
    recommended_needle: str
    tension_bias: str
    topping_required: bool
    stretch_level: str


FABRIC_PRESETS: Dict[str, FabricPreset] = {
    "cotton_pique": FabricPreset(
        id="cotton_pique",
        label="Cotton Pique / Polo Knit",
        category="Knits",
        description="Textured knit mesh for polo shirts. Stretches moderately.",
        pull_compensation_mm=0.35,
        push_compensation_mm=0.10,
        recommended_density_mm=0.38,
        recommended_stitch_length_mm=4.0,
        min_stitch_length_mm=0.8,
        recommended_underlay="auto",
        recommended_stabilizer="Cut-away 2.5 oz + PolyMesh backing",
        recommended_needle="75/11 Ballpoint / SES",
        tension_bias="balanced",
        topping_required=False,
        stretch_level="medium",
    ),
    "denim_twill": FabricPreset(
        id="denim_twill",
        label="Denim / Heavy Twill / Caps",
        category="Heavy Woven",
        description="Dense diagonal woven fabric. High firmness and low stretch.",
        pull_compensation_mm=0.20,
        push_compensation_mm=0.05,
        recommended_density_mm=0.35,
        recommended_stitch_length_mm=4.0,
        min_stitch_length_mm=1.0,
        recommended_underlay="perpendicular",
        recommended_stabilizer="Firm Tear-away 2.0 oz (or Cap Backing)",
        recommended_needle="80/12 Sharp / RG",
        tension_bias="firm",
        topping_required=False,
        stretch_level="low",
    ),
    "silk_rayon": FabricPreset(
        id="silk_rayon",
        label="Silk / Rayon / Sheer",
        category="Delicate / Filament",
        description="Ultra-fine, slippery filament weave. Prone to severe puckering.",
        pull_compensation_mm=0.25,
        push_compensation_mm=0.05,
        recommended_density_mm=0.42,
        recommended_stitch_length_mm=3.5,
        min_stitch_length_mm=0.8,
        recommended_underlay="edge-run",
        recommended_stabilizer="Soft Cut-away 1.5 oz or Light Tear-away",
        recommended_needle="65/9 or 70/10 Microtex Sharp",
        tension_bias="light",
        topping_required=False,
        stretch_level="low",
    ),
    "fleece_terry": FabricPreset(
        id="fleece_terry",
        label="Fleece / Terry Towel / High Pile",
        category="High Loft / Pile",
        description="Thick loft/pile that buries embroidery stitches. Needs double zigzag underlay.",
        pull_compensation_mm=0.45,
        push_compensation_mm=0.15,
        recommended_density_mm=0.34,
        recommended_stitch_length_mm=4.2,
        min_stitch_length_mm=1.0,
        recommended_underlay="double-zigzag",
        recommended_stabilizer="Water-Soluble Topping (Solvy) + Medium Cut-away 3.0 oz backing",
        recommended_needle="75/11 Ballpoint",
        tension_bias="balanced",
        topping_required=True,
        stretch_level="medium",
    ),
    "leather_vinyl": FabricPreset(
        id="leather_vinyl",
        label="Leather / Vinyl / Non-Woven",
        category="Membrane / Non-Woven",
        description="Dense membrane. Needle punctures are permanent cuts; minimal underlay.",
        pull_compensation_mm=0.15,
        push_compensation_mm=0.00,
        recommended_density_mm=0.45,
        recommended_stitch_length_mm=4.5,
        min_stitch_length_mm=1.2,
        recommended_underlay="none",
        recommended_stabilizer="Medium Tear-away (no adhesive)",
        recommended_needle="80/12 Leather / Wedge / LL",
        tension_bias="relaxed",
        topping_required=False,
        stretch_level="none",
    ),
    "poly_spandex": FabricPreset(
        id="poly_spandex",
        label="Poly Spandex / Performance Lycra",
        category="Activewear / Elastic",
        description="4-way high elastic stretch. Extreme pull distortion requires maximum compensation.",
        pull_compensation_mm=0.50,
        push_compensation_mm=0.20,
        recommended_density_mm=0.36,
        recommended_stitch_length_mm=3.8,
        min_stitch_length_mm=0.8,
        recommended_underlay="double-zigzag",
        recommended_stabilizer="Heavy Cut-away (No-Show PolyMesh)",
        recommended_needle="70/10 or 75/11 Ballpoint / Stretch",
        tension_bias="balanced",
        topping_required=False,
        stretch_level="extreme",
    ),
}


class FabricEngine:
    """Provides fabric compensation presets."""
    @classmethod
    def get_preset(cls, fabric_id: str) -> FabricPreset:
        return FABRIC_PRESETS.get(fabric_id, FABRIC_PRESETS["cotton_pique"])
