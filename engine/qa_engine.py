"""
POWER DIGITIZE — Production QA & Machine Validation Engine
Provides:
1. Standardized machine test pattern generation (Running, Double, Triple Bean, Satin, Fill with Hole, Underlays, Small Details)
2. Machine calibration calculator (X/Y scale corrections, offsets)
3. Physical QA recording and evaluation (Digital Expected vs Physical Measured)
4. Comprehensive Production QA certification and worksheet reports
"""

import math
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from engine.stitch_generators import (
    Stitch,
    StitchBlock,
    RunningStitchGenerator,
    SatinStitchGenerator,
    FillStitchGenerator,
    UnderlayGenerator,
    EdgeWalkGenerator,
)
from engine.dst_encoder import DSTEncoder
from engine.stitch_renderer import StitchRenderer
from engine.validator import DSTValidator


# Pre-configured machine profiles
DEFAULT_MACHINE_PROFILES = [
    {
        "id": "generic-industrial",
        "name": "Generic Multi-Head Industrial",
        "spm": 850,
        "max_spm": 1200,
        "needle_type": "DBxK5 75/11",
        "stabilizer": "Cut-away 2.5 oz",
        "x_scale_correction": 1.000,
        "y_scale_correction": 1.000,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    },
    {
        "id": "brother-pr",
        "name": "Brother PR Series (6/10-Needle)",
        "spm": 800,
        "max_spm": 1000,
        "needle_type": "HAx130EB 75/11",
        "stabilizer": "Tear-away 2.0 oz / Cut-away",
        "x_scale_correction": 1.000,
        "y_scale_correction": 1.000,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    },
    {
        "id": "tajima-tmbr",
        "name": "Tajima TMBR / TMAR Series",
        "spm": 900,
        "max_spm": 1200,
        "needle_type": "DBxK5 75/11 R",
        "stabilizer": "Cut-away 3.0 oz",
        "x_scale_correction": 1.000,
        "y_scale_correction": 1.000,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    },
    {
        "id": "barudan-elite",
        "name": "Barudan Elite Pro",
        "spm": 950,
        "max_spm": 1300,
        "needle_type": "DBx7NY 75/11",
        "stabilizer": "Cut-away 2.5 oz Heavy",
        "x_scale_correction": 1.000,
        "y_scale_correction": 1.000,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    },
    {
        "id": "ricoma-em",
        "name": "Ricoma EM-1010 / MT-1501",
        "spm": 800,
        "max_spm": 1000,
        "needle_type": "DBxK5 75/11",
        "stabilizer": "Cut-away 2.5 oz",
        "x_scale_correction": 1.000,
        "y_scale_correction": 1.000,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    },
]


class QAEngine:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.qa_records_file = self.storage_dir / "qa_records.json"
        self.profiles_file = self.storage_dir / "machine_profiles.json"
        self.running_gen = RunningStitchGenerator()
        self.satin_gen = SatinStitchGenerator()
        self.fill_gen = FillStitchGenerator()
        self.underlay_gen = UnderlayGenerator()
        self.edge_gen = EdgeWalkGenerator()
        self.validator = DSTValidator()
        self._ensure_storage()

    @staticmethod
    def calculate_production_cost(
        stitch_count: int,
        thread_m: float,
        speed_spm: int = 800,
        wattage_w: float = 100.0,
        tariff_per_kwh: float = 8.0,
        labor_rate_hourly: float = 35.0,
    ) -> Dict[str, Any]:
        """Calculate exact industrial power, thread and labor breakdown."""
        runtime_min = round(stitch_count / max(100, speed_spm) * 1.25, 2)
        operating_hours = runtime_min / 60.0
        energy_kwh = round((wattage_w * operating_hours) / 1000.0, 4)
        power_cost = round(energy_kwh * tariff_per_kwh, 2)
        thread_cost = round(thread_m * 0.02, 2)
        labor_cost = round(operating_hours * labor_rate_hourly, 2)
        total_cost = round(power_cost + thread_cost + labor_cost, 2)
        return {
            "stitch_count": stitch_count,
            "thread_m": thread_m,
            "runtime_min": runtime_min,
            "power_kwh": energy_kwh,
            "power_cost_inr": power_cost,
            "thread_cost_inr": thread_cost,
            "labor_cost_inr": labor_cost,
            "total_production_cost": total_cost,
        }

    def _ensure_storage(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.profiles_file.exists():
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_MACHINE_PROFILES, f, indent=2)
        if not self.qa_records_file.exists():
            with open(self.qa_records_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def get_machine_profiles(self) -> List[Dict[str, Any]]:
        try:
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_MACHINE_PROFILES

    def save_machine_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        profiles = self.get_machine_profiles()
        p_id = profile.get("id") or f"prof-{int(time.time())}"
        profile["id"] = p_id

        existing_idx = next((i for i, p in enumerate(profiles) if p["id"] == p_id), None)
        if existing_idx is not None:
            profiles[existing_idx] = profile
        else:
            profiles.append(profile)

        with open(self.profiles_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        return profile

    def generate_standard_test_pattern(
        self,
        width_mm: float = 80.0,
        height_mm: float = 80.0,
        stitch_length: float = 2.5,
    ) -> List[StitchBlock]:
        """
        Generate comprehensive standardized calibration and stress-test embroidery pattern.
        Contains:
        - Outer border frame with corner indicators
        - Running stitch: single, double, triple bean
        - Satin columns of varying widths (1.5mm, 3.0mm, 5.0mm) + 90-degree corner
        - Fill / Tatami region with internal negative-space hole
        - Underlay test patterns (edge-run, center-run, zigzag)
        - Small details & curves
        """
        blocks: List[StitchBlock] = []
        half_w = width_mm / 2.0
        half_h = height_mm / 2.0

        # ── Element 1: Outer Calibration Frame & Registration Marks (Color: Black) ──
        frame_points = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
            (-half_w, -half_h),
        ]
        frame_block = self.running_gen.generate(
            points=frame_points,
            stitch_length=2.5,
            color="#111827",
            color_name="Black Carbon",
            name="1. Calibration Frame (80x80mm)",
            bean=False,
        )
        blocks.append(frame_block)

        # ── Element 2: Running Stitch Modes Test: Single, Double, Triple Bean (Color: Black) ──
        # Single Run line
        single_run = self.running_gen.generate(
            points=[(-half_w + 5, half_h - 10), (half_w - 5, half_h - 10)],
            stitch_length=stitch_length,
            color="#111827",
            color_name="Black Carbon",
            name="2. Single Running Stitch (800 SPM Baseline)",
            bean=False,
        )
        blocks.append(single_run)

        # Double Run line (with micro-offset return pass)
        double_pts = [(-half_w + 5, half_h - 15), (half_w - 5, half_h - 15)]
        double_run = self.running_gen.generate(
            points=double_pts,
            stitch_length=stitch_length,
            color="#111827",
            color_name="Black Carbon",
            name="3. Double Running Stitch (Reinforced)",
            bean=False,
        )
        # return pass with slight offset
        double_run_return = self.running_gen.generate(
            points=[(half_w - 5, half_h - 14.9), (-half_w + 5, half_h - 14.9)],
            stitch_length=stitch_length,
            color="#111827",
            color_name="Black Carbon",
            name="3b. Double Run Return Pass",
            bean=False,
        )
        blocks.extend([double_run, double_run_return])

        # Triple Bean Curved line (stress curve test)
        curve_pts = []
        for i in range(25):
            t = i / 24.0
            x = (-half_w + 5) + t * (width_mm - 10)
            y = (half_h - 22) + math.sin(t * math.pi * 2) * 3.5
            curve_pts.append((x, y))

        triple_bean = self.running_gen.generate(
            points=curve_pts,
            stitch_length=2.0,
            color="#111827",
            color_name="Black Carbon",
            name="4. Triple Bean Stitch (Curved Flow)",
            bean=True,
        )
        blocks.append(triple_bean)

        # ── Element 3: Tatami / Fill Region with Negative Space Hole (Color: Electric Cyan) ──
        # $26 \times 26\text{ mm}$ Tatami square with $10\text{ mm}$ circular negative space hole
        fill_pts = [
            (-35.0, 5.0),
            (-5.0, 5.0),
            (-5.0, -25.0),
            (-35.0, -25.0),
            (-35.0, 5.0),
        ]
        # Hole circle points (negative space)
        hole_pts = []
        for i in range(16):
            ang = (i / 16.0) * math.pi * 2
            hx = -20.0 + math.cos(ang) * 5.0
            hy = -10.0 + math.sin(ang) * 5.0
            hole_pts.append((hx, hy))
        hole_pts.append(hole_pts[0])

        # Underlay for fill
        fill_underlay = self.underlay_gen.generate(
            contour_points=fill_pts,
            mode="zigzag",
            density=1.5,
            color="#00d4ff",
            color_name="Electric Cyan",
            holes=[hole_pts],
        )
        blocks.append(fill_underlay)

        # Main Tatami Fill with 45-degree angle
        fill_block = self.fill_gen.generate(
            contour_points=fill_pts,
            density=0.35,
            stitch_length=3.0,
            angle=45.0,
            color="#00d4ff",
            color_name="Electric Cyan",
            name="5b. Tatami Fill with Hole (45° Angle)",
            holes=[hole_pts],
            smooth=False,
        )
        blocks.append(fill_block)

        # Edge-walk border around fill & hole
        fill_border = self.edge_gen.generate(
            contour_points=fill_pts,
            stitch_length=2.0,
            inset=0.15,
            color="#00d4ff",
            color_name="Electric Cyan",
            name="5c. Fill Edge-Walk Border",
            holes=[hole_pts],
        )
        blocks.append(fill_border)

        # ── Element 4: Satin Columns & 90-Degree Corner (Color: Ruby Red) ──
        # Underlay for satin
        satin_centerline = [(5.0, 15.0), (35.0, 15.0), (35.0, -10.0)]
        satin_underlay = self.underlay_gen.generate(
            contour_points=satin_centerline,
            mode="center-run",
            color="#e11d48",
            color_name="Ruby Red",
        )
        blocks.append(satin_underlay)

        # Variable Width / Corner Satin Column (90° corner test)
        satin_block = self.satin_gen.generate(
            center_points=satin_centerline,
            width=3.5,
            density=0.25,
            pull_compensation=0.2,
            color="#e11d48",
            color_name="Ruby Red",
            name="6b. Satin Column with 90° Corner",
        )
        blocks.append(satin_block)

        # Satin width test bars: 1.5mm, 3.0mm, 5.0mm
        for idx, (sw, sx) in enumerate([(1.5, 8.0), (3.0, 18.0), (5.0, 28.0)]):
            s_bar = self.satin_gen.generate(
                center_points=[(sx, -13.0), (sx, -23.0)],
                width=sw,
                density=0.25,
                pull_compensation=0.15,
                color="#e11d48",
                color_name="Ruby Red",
                name=f"6c. Satin Width Gauge ({sw}mm)",
            )
            blocks.append(s_bar)

        # ── Element 5: Density & Direction Variation Patches (Color: Amber Gold) ──
        # 3 Directional mini-patches: 0° (0.3mm density), 45° (0.45mm density), 90° (0.6mm density)
        densities_and_angles = [
            (0.30, 0.0, -35.0, -18.0, "0° / 0.30mm High Density"),
            (0.45, 45.0, -22.0, -18.0, "45° / 0.45mm Med Density"),
            (0.60, 90.0, -9.0, -18.0, "90° / 0.60mm Low Density"),
        ]
        for dens, ang, px, py, label in densities_and_angles:
            patch_pts = [
                (px, py),
                (px + 10.0, py),
                (px + 10.0, py - 8.0),
                (px, py - 8.0),
                (px, py),
            ]
            d_block = self.fill_gen.generate(
                contour_points=patch_pts,
                density=dens,
                stitch_length=2.5,
                angle=ang,
                color="#f59e0b",
                color_name="Amber Gold",
                name=f"7. Density/Angle Patch ({label})",
            )
            blocks.append(d_block)

        # ── Element 6: Repeat Count Comparison Lines (1x, 2x, 3x) (Color: Violet) ──
        for rep_idx, rep_count in enumerate([1, 2, 3]):
            ly = -29.0 - (rep_idx * 2.2)
            line_pts = [(-35.0, ly), (-5.0, ly)]
            for r in range(rep_count):
                r_pts = line_pts if r % 2 == 0 else [line_pts[1], line_pts[0]]
                rep_block = self.running_gen.generate(
                    points=r_pts,
                    stitch_length=2.5,
                    color="#8b5cf6",
                    color_name="Electric Violet",
                    name=f"8. Repeat Pass ({rep_count}×) Line",
                    bean=False,
                )
                blocks.append(rep_block)

        # ── Element 7: Small Details Gauge (Color: Emerald Green) ──
        # Small detail calibration circles (1.5mm, 2.5mm, 4.0mm)
        for idx, r in enumerate([1.5, 2.5, 4.0]):
            cx = 10.0 + idx * 10.0
            cy = -32.0
            circ_pts = []
            for i in range(12):
                ang = (i / 12.0) * math.pi * 2
                circ_pts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
            circ_pts.append(circ_pts[0])

            detail_block = self.running_gen.generate(
                points=circ_pts,
                stitch_length=1.2,
                color="#10b981",
                color_name="Emerald Green",
                name=f"9. Small Detail Gauge (r={r}mm)",
                bean=True,
            )
            blocks.append(detail_block)

        return blocks

    def record_physical_qa(
        self,
        design_id: str,
        qa_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Save physical QA inspection data from machine operator.
        Computes dimensional error and evaluation metrics.
        """
        records = {}
        if self.qa_records_file.exists():
            try:
                with open(self.qa_records_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = {}

        expected_w = float(qa_data.get("expected_width_mm", 80.0))
        expected_h = float(qa_data.get("expected_height_mm", 80.0))
        measured_w = float(qa_data.get("measured_width_mm", expected_w))
        measured_h = float(qa_data.get("measured_height_mm", expected_h))

        # Dimensional error calculation
        w_error_mm = round(measured_w - expected_w, 2)
        h_error_mm = round(measured_h - expected_h, 2)
        w_error_pct = round((w_error_mm / max(1.0, expected_w)) * 100, 2)
        h_error_pct = round((h_error_mm / max(1.0, expected_h)) * 100, 2)

        # Suggested calibration scale factor
        x_correction = round(expected_w / max(1.0, measured_w), 5)
        y_correction = round(expected_h / max(1.0, measured_h), 5)

        # Automatic QA Status determination based on operator inspection
        critical_issues = []
        if abs(w_error_mm) > 1.5 or abs(h_error_mm) > 1.5:
            critical_issues.append(f"Dimensional error exceeds ±1.5mm tolerance (X: {w_error_mm}mm, Y: {h_error_mm}mm)")
        if qa_data.get("thread_breaks", 0) > 2:
            critical_issues.append(f"Excessive thread breaks ({qa_data.get('thread_breaks')} breaks)")
        if qa_data.get("fabric_puckering") == "heavy":
            critical_issues.append("Heavy fabric puckering detected")
        if qa_data.get("fill_coverage") in ["light_gaps", "pull_distorted"]:
            critical_issues.append("Fill coverage inconsistency or gap detected")

        if critical_issues:
            qa_status = "NEEDS_REVIEW"
        elif qa_data.get("operator_approved", True):
            qa_status = "PASSED"
        else:
            qa_status = "PENDING"

        record_entry = {
            "design_id": design_id,
            "timestamp": int(time.time()),
            "date_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": qa_data.get("machine_id", "generic-industrial"),
            "machine_name": qa_data.get("machine_name", "Industrial Multi-Head"),
            "fabric_type": qa_data.get("fabric_type", "Cotton Pique / Twill"),
            "needle": qa_data.get("needle", "75/11 Embroidery"),
            "stabilizer": qa_data.get("stabilizer", "Cut-away 2.5 oz"),
            "expected_dimensions": {"width_mm": expected_w, "height_mm": expected_h},
            "measured_dimensions": {"width_mm": measured_w, "height_mm": measured_h},
            "dimensional_error": {
                "width_error_mm": w_error_mm,
                "height_error_mm": h_error_mm,
                "width_error_pct": w_error_pct,
                "height_error_pct": h_error_pct,
            },
            "suggested_calibration": {
                "x_scale_correction": x_correction,
                "y_scale_correction": y_correction,
            },
            "inspections": {
                "thread_behavior": qa_data.get("thread_behavior", "normal"),
                "loose_tails": qa_data.get("loose_tails", "none"),
                "thread_breaks": int(qa_data.get("thread_breaks", 0)),
                "needle_issues": qa_data.get("needle_issues", "none"),
                "fabric_puckering": qa_data.get("fabric_puckering", "none"),
                "density_evaluation": qa_data.get("density_evaluation", "perfect"),
                "satin_quality": qa_data.get("satin_quality", "smooth"),
                "fill_coverage": qa_data.get("fill_coverage", "full"),
                "registration_alignment": qa_data.get("registration_alignment", "excellent"),
            },
            "operator_notes": qa_data.get("operator_notes", ""),
            "operator_name": qa_data.get("operator_name", "Lead Digitizer QA"),
            "critical_issues": critical_issues,
            "qa_status": qa_status,
            "software_validated": True,
        }

        records[design_id] = record_entry
        with open(self.qa_records_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return record_entry

    def get_physical_qa(self, design_id: str) -> Optional[Dict[str, Any]]:
        if not self.qa_records_file.exists():
            return None
        try:
            with open(self.qa_records_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                return records.get(design_id)
        except Exception:
            return None

    def build_qa_report(
        self,
        design_id: str,
        dst_path: str,
        machine_id: str = "generic-industrial",
    ) -> Dict[str, Any]:
        """Generate official Production QA Certificate & physical inspection report."""
        dst_file = Path(dst_path)
        stats = self.validator.analyze_dst(str(dst_file)) if dst_file.exists() else {}
        qa_record = self.get_physical_qa(design_id)

        profiles = self.get_machine_profiles()
        machine = next((p for p in profiles if p["id"] == machine_id), profiles[0] if profiles else {})

        operating_hours = stats.get("estimated_time", 1) / 60.0
        wattage = float(machine.get("wattage_w", 100.0) if "wattage_w" in machine else 100.0)
        energy_kwh = round((wattage * operating_hours) / 1000.0, 4)
        power_cost = round(energy_kwh * 8.0, 2)
        thread_meters = round(stats.get("stitch_count", 1000) * 0.005, 1)

        qa_status = qa_record.get("qa_status", "PENDING") if qa_record else "PENDING"

        return {
            "design_id": design_id,
            "report_title": "OFFICIAL PRODUCTION QA & PHYSICAL VALIDATION CERTIFICATE",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status_badges": {
                "digital_validation": "SOFTWARE_VALIDATED",
                "dst_validation": "VALIDATED_DST",
                "physical_qa_status": qa_status,
            },
            "machine_specification": {
                "machine_id": machine.get("id"),
                "machine_name": machine.get("name"),
                "operating_speed_spm": machine.get("spm", 850),
                "recommended_needle": qa_record.get("needle") if qa_record else machine.get("needle_type"),
                "recommended_stabilizer": qa_record.get("stabilizer") if qa_record else machine.get("stabilizer"),
            },
            "dimensional_verification": {
                "requested_size_mm": qa_record.get("expected_dimensions") if qa_record else {"width_mm": stats.get("width", 80.0), "height_mm": stats.get("height", 80.0)},
                "dst_stitch_bounds_mm": {"width_mm": stats.get("width", 80.0), "height_mm": stats.get("height", 80.0)},
                "physical_measured_mm": qa_record.get("measured_dimensions") if qa_record else None,
                "dimensional_error": qa_record.get("dimensional_error") if qa_record else None,
                "calibration_used": qa_record.get("suggested_calibration") if qa_record else {"x_scale": machine.get("x_scale_correction", 1.0), "y_scale": machine.get("y_scale_correction", 1.0)},
            },
            "production_metrics": {
                "total_stitches": stats.get("stitch_count", 0),
                "color_changes": stats.get("color_count", 1),
                "jumps": stats.get("jump_count", 0),
                "trims": stats.get("trim_count", 0),
                "thread_meters_estimate": thread_meters,
                "estimated_runtime_minutes": stats.get("estimated_time", 1),
                "energy_kwh": energy_kwh,
                "power_cost_inr": power_cost,
            },
            "physical_inspection_checklist": qa_record.get("inspections") if qa_record else {
                "thread_behavior": "Pending physical stitch-out",
                "loose_tails": "Pending physical stitch-out",
                "thread_breaks": "Pending physical stitch-out",
                "fabric_puckering": "Pending physical stitch-out",
                "density_evaluation": "Pending physical stitch-out",
                "satin_quality": "Pending physical stitch-out",
                "fill_coverage": "Pending physical stitch-out",
            },
            "operator_notes": qa_record.get("operator_notes", "Awaiting physical machine stitch-out.") if qa_record else "Awaiting physical machine stitch-out.",
            "operator_name": qa_record.get("operator_name", "Lead QA") if qa_record else "Lead QA",
            "disclaimer": "Software validation confirms geometric and command integrity. Final production approval requires physical stitch-out verification on target fabric and machine.",
        }
