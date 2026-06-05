"""Validate the Python BVH HIERARCHY against a reference .bvh from the reference engine.

Usage:  python validation/validate_hierarchy.py [reference.bvh] [model.vrm]
Defaults: motion.bvh + cj.vrm in the repo root.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.export.bvh import build_hierarchy, hierarchy_text, js_num  # noqa: E402
from mocapy.vrm.skeleton import load_skeleton  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
def ref_hierarchy_block(bvh_path: Path) -> str:
    lines = []
    for line in bvh_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "MOTION":
            break
        lines.append(line.rstrip("\r"))
    return "\n".join(lines)


def parse_offsets(block: str) -> list[tuple[str, list[float]]]:
    """Return [(joint_name, offset)] in document order (root/joint/end-site)."""
    out = []
    name_stack = []
    pending = None
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            pending = s.split(None, 1)[1]
        elif s == "End Site":
            pending = name_stack[-1] + "/EndSite" if name_stack else "EndSite"
        elif s.startswith("OFFSET"):
            vals = [float(v) for v in s.split()[1:]]
            out.append((pending, vals))
            if not pending.endswith("/EndSite"):
                name_stack.append(pending)
        elif s == "}":
            if name_stack:
                name_stack.pop()
    return out


def main() -> int:
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURES / "motion.bvh"
    vrm_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ASSETS / "cj.vrm"

    skel = load_skeleton(vrm_path)
    mine = hierarchy_text(skel)
    ref = ref_hierarchy_block(ref_path)

    if mine == ref:
        print("EXACT MATCH: HIERARCHY is byte-identical to", ref_path.name)
        return 0

    print("Not byte-identical — comparing structure and numerics.\n")
    ref_off = parse_offsets(ref)
    my_off = parse_offsets(mine)

    print(f"joint count: mine={len(my_off)} ref={len(ref_off)}")
    if [n for n, _ in my_off] != [n for n, _ in ref_off]:
        print("!! joint name/order differs")
        for i, (a, b) in enumerate(zip(my_off, ref_off)):
            if a[0] != b[0]:
                print(f"   [{i}] mine={a[0]!r} ref={b[0]!r}")
        return 1

    max_abs = 0.0
    worst = None
    for (name, a), (_, b) in zip(my_off, ref_off):
        d = np.abs(np.array(a) - np.array(b))
        if d.max() > max_abs:
            max_abs = float(d.max())
            worst = (name, a, b)
    print(f"max abs offset error: {max_abs:.3e}")
    if worst:
        print(f"worst joint: {worst[0]}\n   mine={worst[1]}\n   ref ={worst[2]}")

    # Show the first few literal line differences for formatting insight.
    ml, rl = mine.splitlines(), ref.splitlines()
    shown = 0
    for i in range(min(len(ml), len(rl))):
        if ml[i] != rl[i]:
            print(f"\nline {i} differs:\n  mine: {ml[i]}\n  ref : {rl[i]}")
            shown += 1
            if shown >= 8:
                break
    return 0 if max_abs < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
