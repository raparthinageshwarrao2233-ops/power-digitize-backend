"""
POWER DIGITIZE — Embroidery Object Graph
Manages stable object identities, parent-child relationships,
exclusion masks, and stitch region computation.

Responsibilities:
- Assign stable object IDs to every contour
- Determine parent-child containment (geometric)
- Compute exclusion masks so parent fills don't cover child details
- Provide stitch_region polygons for fill generation
- Track stitch blocks per object for post-hoc analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


@dataclass
class EmbroideryObject:
    """A single embroidery object with full identity and metadata."""

    id: str
    source_contour_id: int          # links back to image_analyzer contour
    geometry: Any                    # Shapely Polygon in embroidery-mm coords
    stitch_region: Any               # geometry minus exclusion masks
    color_hex: str
    color_name: str
    stitch_type: str                 # fill, satin, running, bean, micro_dot
    fill_angle: float = 0.0
    thinness: float = 0.5
    area_px: float = 0.0
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    exclusion_polygons: List[Any] = field(default_factory=list)
    holes: List[List[Tuple[float, float]]] = field(default_factory=list)
    contour_points: List[Tuple[float, float]] = field(default_factory=list)
    centerline_points: List[Tuple[float, float]] = field(default_factory=list)
    stitch_block: Optional[Any] = None   # StitchBlock after generation
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    layer: int = 2                   # 1=underlay, 2=fill, 3=satin, 4=border, 5=detail
    is_protected: bool = False
    semantic_tag: str = "REGION"
    importance_score: float = 0.5
    effective_width: float = 10.0

    @property
    def has_children(self) -> bool:
        return len(self.child_ids) > 0

    @property
    def has_exclusion_mask(self) -> bool:
        return len(self.exclusion_polygons) > 0


class ObjectGraph:
    """
    Registry of embroidery objects with parent-child relationships
    and exclusion-mask computation.

    Usage:
        graph = ObjectGraph()
        objects = graph.build_from_contours(contours, color_map, to_emb, scale, w, h)
        # objects now have .exclusion_polygons for parent fills
    """

    def __init__(self):
        self.objects: Dict[str, EmbroideryObject] = {}
        self._id_counter = 0

    # ── ID allocation ──

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"obj_{self._id_counter:04d}"

    # ── Build from image-analyzer contours ──

    def build_from_contours(
        self,
        contours: List[Dict[str, Any]],
        color_map: Dict[int, Dict],
        to_emb: Callable,
        scale: float,
        img_w: int,
        img_h: int,
    ) -> List[EmbroideryObject]:
        """
        Build the object graph from image analyzer contours.

        1. Create EmbroideryObject for every valid contour.
        2. Determine parent-child by geometric containment.
        3. Compute exclusion masks for parent fills.
        """
        valid_objects: List[EmbroideryObject] = []

        for c in contours:
            if c.get("is_hole"):
                continue
            has_centerline = bool(c.get("centerline_points") and len(c.get("centerline_points")) >= 2)
            if not has_centerline and c.get("stitch_type") == "fill" and c["area"] < 25:
                continue

            # Filter cropped edge artifacts
            bx, by, bw, bh = c["bbox"]["x"], c["bbox"]["y"], c["bbox"]["w"], c["bbox"]["h"]
            if (bx + bw >= img_w - 4 and bw < 25) or (bx <= 2 and bw < 10) or (by + bh >= img_h - 4 and bh < 20):
                continue

            obj_id = self._next_id()
            default_color = {"hex": "#000000", "name": "Black"}
            color_info = color_map.get(c["id"], default_color)

            # Convert to embroidery coordinates
            emb_points = [to_emb(p) for p in c["contour_points"]]
            emb_holes = [
                [to_emb(p) for p in hole]
                for hole in c.get("holes", [])
            ]

            # Build Shapely geometry
            try:
                if len(emb_points) < 3:
                    continue
                hole_arrays = [h for h in emb_holes if len(h) >= 3]
                poly = Polygon(emb_points, hole_arrays)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            except Exception:
                continue

            # Classify stitch type
            has_centerline = bool(c.get("centerline_points") and len(c.get("centerline_points")) >= 2)
            if c["stitch_type"] == "satin" and has_centerline:
                stitch_type = "satin"
                layer = 3
            elif c["stitch_type"] == "running" and has_centerline:
                stitch_type = "running"
                layer = 5
            else:
                stitch_type = "fill"
                layer = 2

            centerline = []
            if has_centerline:
                centerline = [to_emb(p) for p in c["centerline_points"]]

            obj = EmbroideryObject(
                id=obj_id,
                source_contour_id=c["id"],
                geometry=poly,
                stitch_region=poly,
                color_hex=color_info["hex"],
                color_name=color_info["name"],
                stitch_type=stitch_type,
                fill_angle=c.get("fill_angle", 0.0),
                thinness=c.get("thinness", 0.5),
                area_px=c["area"],
                contour_points=emb_points,
                centerline_points=centerline,
                holes=emb_holes,
                layer=layer,
                effective_width=c.get("effective_width", 10.0),
            )
            valid_objects.append(obj)
            self.objects[obj_id] = obj

        # Determine parent-child containment
        self._build_parent_child(valid_objects)

        # Compute exclusion masks for parent fills
        self._compute_exclusion_masks(valid_objects)

        return valid_objects

    # ── Parent-child relationship by geometric containment ──

    def _build_parent_child(self, objects: List[EmbroideryObject]):
        """
        For each object, find the *smallest* enclosing parent.
        A is parent of B iff A's envelope (or geometry) contains B's centroid and A.area > B.area,
        and more than 50 % of B's area intersects A's envelope.
        """
        sorted_objs = sorted(objects, key=lambda o: o.geometry.area, reverse=True)

        def _get_envelope(geom):
            if geom is None or geom.is_empty:
                return None
            if hasattr(geom, "exterior") and geom.exterior is not None:
                return Polygon(geom.exterior)
            elif hasattr(geom, "geoms"):
                parts = [Polygon(g.exterior) for g in geom.geoms if hasattr(g, "exterior") and g.exterior is not None]
                if parts:
                    return unary_union(parts)
            return geom.convex_hull

        for child in sorted_objs:
            if child.parent_id is not None:
                continue
            centroid = child.geometry.centroid

            # Walk from smallest potential parent to largest so we find the
            # immediate (most specific) parent — not a grandparent.
            candidates = sorted(
                [p for p in sorted_objs if p.id != child.id and p.geometry.area > child.geometry.area],
                key=lambda p: p.geometry.area,
            )

            for parent in candidates:
                try:
                    p_geom = parent.geometry
                    p_env = _get_envelope(p_geom)
                    if p_env is None:
                        continue

                    # Check containment in parent geometry or parent exterior envelope (for nested child inside hole)
                    contains_centroid = p_geom.contains(centroid) or p_env.contains(centroid)
                    if not contains_centroid:
                        continue

                    inter_area = p_geom.intersection(child.geometry).area
                    env_inter_area = p_env.intersection(child.geometry).area
                    if inter_area > child.geometry.area * 0.50 or env_inter_area > child.geometry.area * 0.50:
                        child.parent_id = parent.id
                        parent.child_ids.append(child.id)
                        break
                except Exception:
                    continue

    # ── Exclusion masks ──

    def _compute_exclusion_masks(self, objects: List[EmbroideryObject]):
        """
        For each parent *fill*, subtract all child geometries to create the
        stitch_region.  This ensures parent tatami fill never covers child details.
        """
        for obj in objects:
            if obj.stitch_type != "fill" or not obj.child_ids:
                continue

            exclusion_polys = []
            for child_id in obj.child_ids:
                child = self.objects.get(child_id)
                if child is None or child.geometry is None or child.geometry.is_empty:
                    continue
                # Buffer the child shape to leave clean thread-gap separation
                try:
                    buffered = child.geometry.buffer(0.20)
                    if buffered.is_valid and not buffered.is_empty:
                        exclusion_polys.append(buffered)
                except Exception:
                    continue

            if not exclusion_polys:
                continue

            obj.exclusion_polygons = exclusion_polys
            try:
                exclusion_union = unary_union(exclusion_polys)
                stitch_region = obj.geometry.difference(exclusion_union)
                if not stitch_region.is_empty and stitch_region.is_valid:
                    obj.stitch_region = stitch_region
            except Exception:
                pass  # keep original geometry

    # ── Line-art / detail object registration ──

    def add_line_art_object(
        self,
        line_art_obj: Dict[str, Any],
        to_emb: Callable,
    ) -> Optional[EmbroideryObject]:
        """Register a line art or micro-detail object in the graph."""
        obj_id = self._next_id()
        category = line_art_obj.get("category", "STROKE")

        if category == "STROKE":
            pts = line_art_obj.get("points", [])
            if len(pts) < 2:
                return None
            obj = EmbroideryObject(
                id=obj_id,
                source_contour_id=-1,
                geometry=None,
                stitch_region=None,
                color_hex=line_art_obj.get("color_hex", "#141419"),
                color_name=line_art_obj.get("color_name", "Detail Black"),
                stitch_type=line_art_obj.get("stitch_type", "running"),
                contour_points=[to_emb(p) for p in pts],
                layer=5,
                is_protected=line_art_obj.get("is_protected", False),
                semantic_tag=line_art_obj.get("semantic_tag", "DETAIL"),
                importance_score=line_art_obj.get("importance_score", 0.8),
                effective_width=line_art_obj.get("width_mm", 1.0),
            )
        elif category == "MICRO_DOT":
            center = line_art_obj.get("center", (0, 0))
            obj = EmbroideryObject(
                id=obj_id,
                source_contour_id=-1,
                geometry=None,
                stitch_region=None,
                color_hex=line_art_obj.get("color_hex", "#141419"),
                color_name=line_art_obj.get("color_name", "Detail Black"),
                stitch_type="micro_dot",
                contour_points=[to_emb(center)],
                layer=5,
                is_protected=True,
                semantic_tag=line_art_obj.get("semantic_tag", "BINDI_DOT"),
                importance_score=line_art_obj.get("importance_score", 0.95),
                effective_width=line_art_obj.get("diameter_mm", 0.8),
            )
        else:
            return None

        self.objects[obj_id] = obj
        return obj

    # ── Accessors ──

    def get_objects_by_layer(self) -> Dict[int, List[EmbroideryObject]]:
        layers: Dict[int, List[EmbroideryObject]] = {}
        for obj in self.objects.values():
            layers.setdefault(obj.layer, []).append(obj)
        return layers

    def get_object(self, obj_id: str) -> Optional[EmbroideryObject]:
        return self.objects.get(obj_id)

    def get_children(self, obj_id: str) -> List[EmbroideryObject]:
        obj = self.objects.get(obj_id)
        if not obj:
            return []
        return [self.objects[cid] for cid in obj.child_ids if cid in self.objects]

    def get_exclusion_for_contour(self, source_contour_id: int) -> List:
        """Look up exclusion polygons by the original contour ID."""
        for obj in self.objects.values():
            if obj.source_contour_id == source_contour_id:
                return obj.exclusion_polygons
        return []

    # ── Serialisation ──

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the object graph for API responses and inspection."""
        result = {}
        for obj_id, obj in self.objects.items():
            result[obj_id] = {
                "id": obj.id,
                "source_contour_id": obj.source_contour_id,
                "color_hex": obj.color_hex,
                "color_name": obj.color_name,
                "stitch_type": obj.stitch_type,
                "fill_angle": obj.fill_angle,
                "area_px": obj.area_px,
                "parent_id": obj.parent_id,
                "child_ids": obj.child_ids,
                "has_exclusion_mask": obj.has_exclusion_mask,
                "layer": obj.layer,
                "is_protected": obj.is_protected,
                "semantic_tag": obj.semantic_tag,
                "importance_score": obj.importance_score,
                "stitch_count": obj.stitch_block.stitch_count if obj.stitch_block else 0,
                "start_point": list(obj.start_point) if obj.start_point else None,
                "end_point": list(obj.end_point) if obj.end_point else None,
            }
        return result
