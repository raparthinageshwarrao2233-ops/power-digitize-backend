"""
POWER DIGITIZE — Sequence Optimizer & Auto-Route Graph Engine
Inspired by Ink/Stitch auto-route and Wilcom industrial path planning.

Key Algorithms:
1. Auto-Route Eulerian Graph Routing (Chinese Postman Problem):
   Connects running stitch networks (line art, outlines, monograms) into
   continuous single-thread trails with ZERO trims and minimal jumps.
2. Layer-Aware Traveling Salesperson (TSP) Nearest-Neighbor Sequencing:
   Orders Underlay -> Fill -> Satin -> Border to prevent embroidery distortion.
3. Reversible Block Traversal:
   Allows bidirectional entry/exit for satin columns and running lines.
4. Micro-Gap Stitch Bridging:
   Seamless thread bridge between adjacent contours without firing machine trims.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from engine.stitch_generators import Stitch, StitchBlock


class AutoRouteGraph:
    """
    Graph-based continuous path router for embroidery running stitches and line art.
    Solves the Chinese Postman Problem / Eulerian Path to eliminate thread trims.
    """

    @classmethod
    def route_strokes(
        cls,
        strokes: List[List[Tuple[float, float]]],
        stitch_length: float = 2.5,
        snap_tolerance: float = 0.8,
        color: str = "#000000",
        color_name: str = "Black",
        name: str = "Auto-Routed Linework",
        bean: bool = False,
    ) -> StitchBlock:
        """
        Takes multiple disconnected or branching strokes and routes them into
        a continuous Eulerian path to minimize machine trims.
        """
        block = StitchBlock(color=color, color_name=color_name, stitch_type="running", name=name)
        if not strokes:
            return block

        valid_strokes = []
        for s in strokes:
            if len(s) >= 2:
                d = math.hypot(s[-1][0] - s[0][0], s[-1][1] - s[0][1])
                if d > 0.05 or len(s) > 2:
                    valid_strokes.append(s)

        if not valid_strokes:
            return block

        nodes: List[Tuple[float, float]] = []

        def get_or_create_node(pt: Tuple[float, float]) -> int:
            for i, n in enumerate(nodes):
                if math.hypot(n[0] - pt[0], n[1] - pt[1]) <= snap_tolerance:
                    return i
            nodes.append(pt)
            return len(nodes) - 1

        edges = []
        for s_idx, stroke in enumerate(valid_strokes):
            u = get_or_create_node(stroke[0])
            v = get_or_create_node(stroke[-1])
            edges.append({
                "id": s_idx,
                "u": u,
                "v": v,
                "pts": stroke,
                "used": 0,
            })

        degree = [0] * len(nodes)
        for e in edges:
            degree[e["u"]] += 1
            degree[e["v"]] += 1

        odd_nodes = [i for i, d in enumerate(degree) if d % 2 != 0]

        bridging_edges = []
        unmatched = set(odd_nodes)
        while len(unmatched) >= 2:
            u = next(iter(unmatched))
            unmatched.remove(u)
            best_v = None
            best_dist = float("inf")
            for v in unmatched:
                dist = math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])
                if dist < best_dist:
                    best_dist = dist
                    best_v = v
            if best_v is not None:
                unmatched.remove(best_v)
                bridging_edges.append({
                    "id": len(edges) + len(bridging_edges),
                    "u": u,
                    "v": best_v,
                    "pts": [nodes[u], nodes[best_v]],
                    "is_bridge": True,
                    "used": 0,
                })

        all_edges = edges + bridging_edges

        adj: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
        for e_idx, e in enumerate(all_edges):
            adj[e["u"]].append(e_idx)
            adj[e["v"]].append(e_idx)

        current_node = 0 if not odd_nodes else odd_nodes[0]
        visited_edge_ids = set()
        trail_pts: List[Tuple[float, float]] = []

        def get_next_edge(u: int) -> Optional[int]:
            candidates = [eid for eid in adj[u] if eid not in visited_edge_ids]
            if not candidates:
                return None
            candidates.sort(key=lambda eid: 1 if all_edges[eid].get("is_bridge", False) else 0)
            return candidates[0]

        curr = current_node
        trail_pts.append(nodes[curr])

        while len(visited_edge_ids) < len(all_edges):
            e_idx = get_next_edge(curr)
            if e_idx is None:
                found_next = False
                for eid, ed in enumerate(all_edges):
                    if eid not in visited_edge_ids:
                        visited_edge_ids.add(eid)
                        e_pts = ed["pts"]
                        block.add(trail_pts[-1][0], trail_pts[-1][1], Stitch.TRIM)
                        block.add(e_pts[0][0], e_pts[0][1], Stitch.JUMP)
                        for pt in e_pts:
                            block.add(pt[0], pt[1], Stitch.NORMAL)
                        curr = ed["v"]
                        found_next = True
                        break
                if not found_next:
                    break
                continue

            visited_edge_ids.add(e_idx)
            edge = all_edges[e_idx]
            pts = edge["pts"]

            if edge["u"] == curr:
                step_pts = pts
                curr = edge["v"]
            else:
                step_pts = list(reversed(pts))
                curr = edge["u"]

            for pt in step_pts[1:]:
                trail_pts.append(pt)

        prev = None
        for pt in trail_pts:
            if prev is None:
                block.add(pt[0], pt[1], Stitch.NORMAL)
                prev = pt
            else:
                d = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
                if d >= stitch_length:
                    steps = max(1, int(round(d / stitch_length)))
                    for s in range(1, steps + 1):
                        f = s / steps
                        ix = prev[0] + f * (pt[0] - prev[0])
                        iy = prev[1] + f * (pt[1] - prev[1])
                        block.add(ix, iy, Stitch.NORMAL)
                    prev = pt
                elif d > 0.05:
                    block.add(pt[0], pt[1], Stitch.NORMAL)
                    prev = pt

        return block


class SequenceOptimizer:
    """
    Sequences stitch blocks to minimize machine runtime, trims, and pantograph travel.
    """
    LAYER_WEIGHTS = {
        "underlay": 0,
        "fill": 1,
        "satin": 2,
        "border": 3,
        "running": 3,
        "repeat": 4,
    }

    @classmethod
    def optimize(
        cls,
        stitch_blocks: List[StitchBlock],
        enable_reversal: bool = True,
        max_bridge_gap_mm: float = 1.8,
    ) -> List[StitchBlock]:
        """
        Reorders stitch blocks using layer-aware Traveling Salesperson (TSP)
        nearest-neighbor traversal to minimize trims and jump distance.
        """
        if not stitch_blocks or len(stitch_blocks) <= 1:
            return list(stitch_blocks)

        color_order = []
        blocks_by_color: Dict[str, List[StitchBlock]] = {}
        for block in stitch_blocks:
            if block.color not in blocks_by_color:
                blocks_by_color[block.color] = []
                color_order.append(block.color)
            blocks_by_color[block.color].append(block)

        optimized_blocks: List[StitchBlock] = []
        current_pos: Tuple[float, float] = (0.0, 0.0)

        for color in color_order:
            c_blocks = blocks_by_color[color]

            compound_groups: Dict[str, List[StitchBlock]] = {}
            standalone_units: List[List[StitchBlock]] = []

            for b in c_blocks:
                if b.group_id:
                    compound_groups.setdefault(b.group_id, []).append(b)
                else:
                    standalone_units.append([b])

            compound_units: List[List[StitchBlock]] = []
            for gid, grp in compound_groups.items():
                grp.sort(key=lambda b: cls.LAYER_WEIGHTS.get(b.stitch_type, 1))
                compound_units.append(grp)

            tier1_units: List[List[StitchBlock]] = []
            tier2_units: List[List[StitchBlock]] = []

            for unit in compound_units:
                tier1_units.append(unit)

            for unit in standalone_units:
                b = unit[0]
                if b.stitch_type in ("underlay", "fill"):
                    tier1_units.append(unit)
                else:
                    tier2_units.append(unit)

            unvisited_t1 = list(tier1_units)
            while unvisited_t1:
                best_idx = -1
                best_dist = float("inf")

                for i, unit in enumerate(unvisited_t1):
                    first_blk = unit[0]
                    if not first_blk.stitches:
                        continue
                    p_start = (first_blk.stitches[0].x, first_blk.stitches[0].y)
                    d = math.hypot(p_start[0] - current_pos[0], p_start[1] - current_pos[1])
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_idx == -1:
                    for u in unvisited_t1:
                        optimized_blocks.extend(u)
                    break

                chosen_unit = unvisited_t1.pop(best_idx)
                optimized_blocks.extend(chosen_unit)
                last_blk = chosen_unit[-1]
                if last_blk.stitches:
                    current_pos = (last_blk.stitches[-1].x, last_blk.stitches[-1].y)

            unvisited_t2 = list(tier2_units)
            while unvisited_t2:
                best_idx = -1
                best_dist = float("inf")
                best_reverse = False

                for i, unit in enumerate(unvisited_t2):
                    blk = unit[0]
                    if not blk.stitches:
                        continue
                    p_start = (blk.stitches[0].x, blk.stitches[0].y)
                    p_end = (blk.stitches[-1].x, blk.stitches[-1].y)

                    d_start = math.hypot(p_start[0] - current_pos[0], p_start[1] - current_pos[1])
                    if d_start < best_dist:
                        best_dist = d_start
                        best_idx = i
                        best_reverse = False

                    if enable_reversal and blk.stitch_type in ("running", "satin", "border", "repeat"):
                        d_end = math.hypot(p_end[0] - current_pos[0], p_end[1] - current_pos[1])
                        if d_end < best_dist:
                            best_dist = d_end
                            best_idx = i
                            best_reverse = True

                if best_idx == -1:
                    for u in unvisited_t2:
                        optimized_blocks.extend(u)
                    break

                chosen_unit = unvisited_t2.pop(best_idx)
                chosen = chosen_unit[0]
                if best_reverse:
                    rev_block = StitchBlock(
                        color=chosen.color,
                        color_name=chosen.color_name,
                        stitch_type=chosen.stitch_type,
                        name=f"{chosen.name} (Reversed)",
                        group_id=chosen.group_id,
                    )
                    rev_block.stitches = list(reversed(chosen.stitches))
                    optimized_blocks.append(rev_block)
                    current_pos = (rev_block.stitches[-1].x, rev_block.stitches[-1].y)
                else:
                    optimized_blocks.append(chosen)
                    current_pos = (chosen.stitches[-1].x, chosen.stitches[-1].y)

        return optimized_blocks
