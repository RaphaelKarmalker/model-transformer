#!/usr/bin/env python3
"""
USD Inspector (standalone, no Isaac Sim)

What it does:
1) Opens a native file picker so you can select a .usd/.usda/.usdc
2) Opens the USD stage
3) Writes bounded outputs (so huge stages don't spam your terminal):
   - tree.txt          : bounded stage tree (by depth + node limit)
   - xforms.txt        : Xform prims (good "manipulable" candidates)
   - meshes.txt        : Mesh prims (reference only)
   - summary.json      : quick counts + basic stats
4) Prints where outputs were written

Requirements:
- Pixar USD Python bindings (module: pxr)
  Install (one of these):
    conda install -c conda-forge usd-core
    pip install usd-core

Run:
  python usd_inspector.py

Options:
  python usd_inspector.py --start /World/Robot --max-depth 6 --max-nodes 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# GUI picker (built-in)
from tkinter import Tk, filedialog

# USD
from pxr import Usd, UsdGeom


# ----------------------------
# Config + Helpers
# ----------------------------
def pick_usd_file(initial_dir: Optional[str] = None) -> Optional[str]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select a USD file",
        initialdir=initial_dir or os.getcwd(),
        filetypes=[
            ("USD files", "*.usd *.usda *.usdc *.usdz"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def open_stage(path: str) -> Usd.Stage:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    return stage


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_in_subtree(prim_path: str, start_path: Optional[str]) -> bool:
    if not start_path:
        return True
    s = start_path.rstrip("/")
    return prim_path == s or prim_path.startswith(s + "/")


def prim_type(prim: Usd.Prim) -> str:
    t = prim.GetTypeName()
    return t if t else "UnknownType"


def sorted_children(prim: Usd.Prim) -> list[Usd.Prim]:
    kids = list(prim.GetChildren())
    kids.sort(key=lambda p: p.GetName())
    return kids


@dataclass
class WalkStats:
    total_prims: int = 0
    active_prims: int = 0
    xform_prims: int = 0
    mesh_prims: int = 0
    max_depth_seen: int = 0


# ----------------------------
# Bounded tree print
# ----------------------------
def build_bounded_tree_lines(
    stage: Usd.Stage,
    start_path: Optional[str],
    max_depth: int,
    max_nodes: int,
    include_types: bool,
) -> Tuple[list[str], WalkStats]:
    stats = WalkStats()

    if start_path:
        start_prim = stage.GetPrimAtPath(start_path)
        if not start_prim or not start_prim.IsValid():
            raise ValueError(f"Start prim not found/invalid: {start_path}")
    else:
        start_prim = stage.GetPseudoRoot()

    lines: list[str] = []
    node_count = 0

    def emit(prim: Usd.Prim, depth: int) -> None:
        nonlocal node_count
        if node_count >= max_nodes:
            return
        if depth > max_depth:
            return

        stats.max_depth_seen = max(stats.max_depth_seen, depth)

        if include_types:
            lines.append(f'{"  " * depth}- {prim.GetPath()} [{prim_type(prim)}]')
        else:
            lines.append(f'{"  " * depth}- {prim.GetPath()}')

        node_count += 1
        if node_count >= max_nodes:
            return

        for child in sorted_children(prim):
            emit(child, depth + 1)
            if node_count >= max_nodes:
                break

    # If pseudo-root, print children at depth 0 (don't print "/")
    if start_prim.GetPath() == "/":
        for child in sorted_children(start_prim):
            emit(child, 0)
            if node_count >= max_nodes:
                break
    else:
        emit(start_prim, 0)

    if node_count >= max_nodes:
        lines.append(f"\n[TRUNCATED] Reached max_nodes={max_nodes}. Increase --max-nodes to see more.")

    # Stage-wide stats (bounded by subtree filter)
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not is_in_subtree(p, start_path):
            continue

        stats.total_prims += 1
        if prim.IsActive():
            stats.active_prims += 1
        if prim.IsA(UsdGeom.Xform):
            stats.xform_prims += 1
        if prim.GetTypeName() == "Mesh":
            stats.mesh_prims += 1

    return lines, stats


# ----------------------------
# Lists for LLM/whitelists
# ----------------------------
def collect_xforms(stage: Usd.Stage, start_path: Optional[str]) -> list[str]:
    out: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        p = str(prim.GetPath())
        if not is_in_subtree(p, start_path):
            continue
        if prim.IsA(UsdGeom.Xform):
            out.append(f"{prim.GetPath()} [{prim_type(prim)}]")
    return out


def collect_meshes(stage: Usd.Stage, start_path: Optional[str]) -> list[str]:
    out: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        p = str(prim.GetPath())
        if not is_in_subtree(p, start_path):
            continue
        if prim.GetTypeName() == "Mesh":
            out.append(str(prim.GetPath()))
    return out


def collect_manipulable_candidates(stage: Usd.Stage, start_path: Optional[str]) -> list[str]:
    """
    Practical heuristic:
    - Prefer Xform prims that either:
      - have at least one Mesh descendant, OR
      - are common "group/object" nodes even if they aren't meshes themselves
    This tends to match "object-level" paths you want to manipulate.
    """
    candidates: list[str] = []

    # Precompute mesh paths for quick descendant check
    mesh_paths = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        p = str(prim.GetPath())
        if not is_in_subtree(p, start_path):
            continue
        if prim.GetTypeName() == "Mesh":
            mesh_paths.append(p)

    mesh_paths.sort()

    def has_mesh_descendant(path: str) -> bool:
        # binary-search-ish prefix check via linear scan is OK for moderate sizes,
        # but mesh count could be huge; still usually fine for one-time script.
        prefix = path.rstrip("/") + "/"
        # Quick: check any mesh starting with prefix
        # Use "starts_with" with early break because mesh_paths sorted
        for mp in mesh_paths:
            if mp.startswith(prefix):
                return True
            if mp > prefix and not mp.startswith(prefix):
                # Because sorted, once we've passed prefix region, we can break
                # (not perfect, but good enough to early break often)
                if mp[: len(prefix)] > prefix:
                    break
        return False

    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        p = str(prim.GetPath())
        if not is_in_subtree(p, start_path):
            continue
        if not prim.IsA(UsdGeom.Xform):
            continue

        # Ignore trivial transform leaves without children
        kids = prim.GetChildren()
        if not kids:
            continue

        if has_mesh_descendant(p):
            candidates.append(p)

    # Deduplicate + sort
    candidates = sorted(set(candidates))
    return candidates


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=str, default=None, help="USD file path. If omitted, opens a file picker.")
    ap.add_argument("--start", type=str, default=None, help="Restrict output to subtree, e.g. /World/Robot")
    ap.add_argument("--max-depth", type=int, default=6, help="Max depth for tree output")
    ap.add_argument("--max-nodes", type=int, default=20000, help="Max nodes for tree output")
    ap.add_argument("--no-types", action="store_true", help="Do not include prim type names in tree output")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: ./usd_inspector_<timestamp>/)")
    ap.add_argument("--top-k-candidates", type=int, default=300, help="Limit manipulable candidates list length")
    args = ap.parse_args()

    usd_path = args.usd
    if not usd_path:
        usd_path = pick_usd_file()
        if not usd_path:
            print("No file selected.")
            return 1

    usd_path = os.path.abspath(usd_path)

    # Output dir
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path.cwd() / f"usd_inspector_{ts}"
    ensure_dir(outdir)

    # Open USD
    try:
        stage = open_stage(usd_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    # Tree + stats
    try:
        tree_lines, stats = build_bounded_tree_lines(
            stage=stage,
            start_path=args.start,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            include_types=not args.no_types,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 3

    # Lists
    xforms = collect_xforms(stage, args.start)
    meshes = collect_meshes(stage, args.start)
    candidates = collect_manipulable_candidates(stage, args.start)
    if args.top_k_candidates and len(candidates) > args.top_k_candidates:
        candidates = candidates[: args.top_k_candidates] + [
            f"... TRUNCATED (showing first {args.top_k_candidates} of many). Increase --top-k-candidates."
        ]

    # Write files
    tree_file = outdir / "tree.txt"
    xforms_file = outdir / "xforms.txt"
    meshes_file = outdir / "meshes.txt"
    cand_file = outdir / "manipulable_candidates.txt"
    summary_file = outdir / "summary.json"

    tree_file.write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    xforms_file.write_text("\n".join(xforms) + "\n", encoding="utf-8")
    meshes_file.write_text("\n".join(meshes) + "\n", encoding="utf-8")
    cand_file.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    summary = {
        "usd_path": usd_path,
        "start_subtree": args.start,
        "tree": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "include_types": (not args.no_types),
            "written_to": str(tree_file),
        },
        "counts_in_subtree": {
            "total_prims": stats.total_prims,
            "active_prims": stats.active_prims,
            "xform_prims": stats.xform_prims,
            "mesh_prims": stats.mesh_prims,
            "max_depth_seen_in_tree_output": stats.max_depth_seen,
        },
        "outputs": {
            "xforms": str(xforms_file),
            "meshes": str(meshes_file),
            "manipulable_candidates": str(cand_file),
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== USD Inspector DONE ===")
    print(f"Selected: {usd_path}")
    if args.start:
        print(f"Subtree:  {args.start}")
    print(f"Outdir:   {outdir}")
    print(f"- Tree:   {tree_file.name}")
    print(f"- Xforms: {xforms_file.name}")
    print(f"- Meshes: {meshes_file.name}")
    print(f"- Cand.:  {cand_file.name}")
    print(f"- Summary:{summary_file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())