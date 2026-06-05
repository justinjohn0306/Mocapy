"""ARKit-52 blendshapes -> MMD morphs (compatible with VRM blendShapeGroups).

Ported from the reference engine's blendshape mapping (min.js ~245135-251005). The function
takes MediaPipe's 52 ARKit-style blendshape weights and returns a dict of MMD-named
morph weights — the same names that appear in golden's `morphKeys` (あ/い/う/え/お,
まばたき, にこり, etc.). The MMD→VRM-expression map handles consumers.

Conventions matching the rest of the rig:
* L/R SWAPPED on blink (character's left eye comes from ARKit-eyeBlinkRight),
  because video is mirrored — same convention as the body side-swap.
* head_pitch_rad biases the "sad" detection (when head looks down, slight shrug is
  not interpreted as sadness — that's just gravity on the face).
"""

from __future__ import annotations

import math


# Standard MMD → VRM 0.x preset map (consumers can remap further).
# `weight_scale` lets a VRM expression amplify or attenuate the MMD-weighted output.
MMD_TO_VRM_PRESET: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "まばたき": "blink", "まばたきL": "blink_l", "まばたきR": "blink_r",
    "ウィンク": "blink_l", "ウィンク右": "blink_r",
    "にこり": "joy", "笑い": "joy",         # both contribute to joy
    "口角上げ": "joy",
    "怒り": "angry",
    "困る": "sorrow",                       # 困る ≈ troubled / sad
    "にやり": "fun",                         # smirk-as-fun
    "びっくり": "fun",                       # surprise → fun fallback (VRM 0 has no `surprised`)
    # 上/下 (look-up/-down) are bone-driven in VRM, not morph-driven; left unmapped.
}


def solve_morphs(blendshapes: dict[str, float],
                 head_pitch_rad: float = 0.0,
                 head_yaw_rad: float = 0.0,
                 *, sensitivity: float = 1.0) -> dict[str, float]:
    """ARKit 52-blendshape dict -> MMD morph weights dict (clamped 0..1).

    blendshapes: e.g. {"jawOpen": 0.42, "mouthSmileLeft": 0.18, ...}
    head_pitch_rad: head pitch (X-euler in radians, positive = looking up). Used
        to detune the "sad" mouth shape when the head is bowed.
    head_yaw_rad: head yaw (Y-euler in radians). Used to attenuate the tongue-out
        (ぺろっ) detection when the head is turned more than 10° from frontal.
    sensitivity: mouth-tracking sensitivity (the reference's Ze.mouth_tracking_sensitivity,
        typically 1.0 — values >1 sharpen the vowel discrimination, <1 soften).
    """
    K = blendshapes
    def k(name: str) -> float:
        v = K.get(name, 0.0)
        return float(v) if v is not None else 0.0

    pi = math.pi
    o = k("jawOpen")
    i_sm = (k("mouthSmileLeft") + k("mouthSmileRight")) * 0.5
    n = (k("mouthUpperUpLeft") + k("mouthUpperUpRight")) * 0.5
    a = (k("mouthLowerDownLeft") + k("mouthLowerDownRight")) * 0.5
    s = k("mouthPucker")
    r = k("mouthFunnel")
    e = k("mouthShrugLower") ** 2
    t_brow_in = k("browInnerUp") ** 2
    brow_up = (k("browOuterUpLeft") + k("browOuterUpRight")) * 0.5
    brow_dn = (k("browDownLeft") + k("browDownRight")) * 0.5

    # ── Sad (T) — head-pitch-biased shrug + brow-down fallback (min.js ~245500) ──
    head_k = head_pitch_rad
    ye = 0.3 * (1.0 - min(head_k / (pi / 12), 1.0) if head_k > 0 else 1.0)
    T = min(max(e - ye, 0.0) / 0.7, 1.0)
    if head_k < 0:
        re = min(abs(head_k) / (pi / 4), 1.0)
        denom = max(1.0 - re * 0.5, 1e-6)
        T = max(T - re * 0.5, 0.0) ** (re * 3.0) / (denom ** (re * 3.0)) if T > 0 else 0.0
    if brow_dn > 0.15:
        T = max(T, ((brow_dn - 0.15) / 0.85) ** 0.4)

    # ── Vowels (a/i/u/e/o) ── min.js ~245700-246300
    J = min(sensitivity, 2.0) / 2.0          # mouth_tracking_sensitivity / 2
    ee = max(sensitivity - 1.0, 0.0) / 2.0
    te = 0.2 + 0.1 * J
    c = d = p = m = u = 0.0
    if o < 0.2:
        if s > 0.1:
            p = s
            if r > 0.1:
                u = (r - 0.1) * 0.5
        else:
            d = min(n + a * 0.5, 1.0) ** (1.0 - 0.5 * ee)
            denom1 = max(o / 0.1, 1e-6)
            m = (min((i_sm + n * 0.5) * (o / 0.1), 1.0)) ** (1.0 - 0.5 * ee)
            denom2 = max(n + a, 0.1)
            re_w = min(i_sm * (o / 0.1) / denom2, 1.0)
            d *= (1.0 - re_w); m *= re_w
        excitement = max(d, p, m, u)
        tt = (1.0 - o / 0.2 * 0.5) - (1.0 - min(excitement, 0.2) / 0.2) * 0.3
        tt = max(1.0 - (1.0 - tt) * J, 0.05)
        c = (o / 0.2) ** tt * te
    else:
        # Jaw-open branch: vowel base scales with how open the jaw is.
        e_open = te + ((o - 0.2) / 0.8) ** (1.0 - 0.5 * ee) * (1.0 - te)
        c = min(e_open + (i_sm + n) / 3.0, 1.0)
        u = min(e_open + r * 0.5, 1.0)
        if s > 0.1:
            re_w = min((s - 0.1) * 5.0, 1.0)
            u *= re_w; c *= (1.0 - re_w)
        else:
            u = 0.0

    # ── Smirk (P) / smile (R) / にこり ── min.js ~249420-249760
    if r > 0.1:
        P = -(r - 0.1) * 0.75
    else:
        P = min(max(i_sm - 0.2, 0.0) + max(n - 0.5, 0.0) * 1.5, 1.0)
        if P > 0.5:
            m *= 1.0 - (P - 0.5) * 0.5
    R = max(P, 0.0) * 1.5 - T
    if R < 0:
        T = -R; R = 0.0
    else:
        T = 0.0; R = min(R * 1.5, 1.0)
    nikori = min(R * 0.8, 0.6)

    # ── Angry (I) ── from mouthLeft/mouthRight horizontal pull (NOT brow_down).
    # `oe = 0.2 + s/2` raises the threshold when the mouth is also puckering.
    oe = 0.2 + s * 0.5
    I = max(k("mouthLeft") - oe, k("mouthRight") - oe, 0.0)

    # ── Final emit-scale fixups (min.js ~249800; we were missing all three) ──
    # `ie = 0.4` is the reference's laugh-emit scale; T is halved; I is suppressed by R.
    R *= 0.4
    T *= 0.5
    I = max(I - R, 0.0)

    # ── Surprise (j) ── min.js ~246900
    sup = brow_up
    if brow_dn > 0.1:
        sup = -(brow_dn - 0.1) / 0.9
    elif t_brow_in > 0.1 and brow_up > 0.1:
        sup = ((t_brow_in - 0.1) / 0.9 + (brow_up - 0.1) / 0.9) * 0.5
    h_blink = (k("eyeBlinkLeft") + k("eyeBlinkRight")) * 0.5
    if sup > 0:
        j = max(sup ** (2.0 / 3) - h_blink ** 1.5 - 0.1, 0.0)
        j = j * j * 2.0 / 3.0
    else:
        j = 0.0

    # ── Eye look up/down (上/下) — gated to suppress brow noise ──
    look_up = max(brow_up - 0.25, 0.0) * (1.0 / 0.75)
    look_dn = max(0.0, k("eyeLookDownLeft") + k("eyeLookDownRight") - 0.2) * 0.5

    # ── Blinks (まばたきL/R) — character/ARKit L/R SWAPPED (mirrored video).
    # Some VRMs only expose ウィンク/ウィンク右; we emit BOTH naming sets so the
    # consumer can pick whichever the rig has. R = laugh damps blinks (per the reference).
    blink_L = max(k("eyeBlinkRight") - R, 0.0)
    blink_R = max(k("eyeBlinkLeft")  - R, 0.0)

    # 口角上げ — "mouth corner up" — the reference: `n - max(e, T, I)` where e = mouthShrugLower²,
    # n = mouthUpperUp mean. So it's mouth-raise minus shrug/sad/angry (not nikori-based).
    mouth_corner_up = max(n - max(e, T, I), 0.0)

    # ── ぺろっ (tongue out, z) ── min.js ~250300. Gated by jaw open + both lower-lip
    # corners pulled down; suppressed by upper-lip rise and head yaw > 10°; amplified
    # by mouth dimple; then a sinusoidal s-curve (ne-branch — the MMD ぺろっ path).
    # g = emotion_weight × emotion_tongue_out scale; at defaults (100%/100%) g = 1.
    g = 1.0
    y_lower = min(k("mouthLowerDownLeft"), k("mouthLowerDownRight"))    # BOTH sides down
    e_upper = max(k("mouthUpperUpLeft"), k("mouthUpperUpRight"))        # EITHER side up
    z = 0.0
    if y_lower > 0.2 + 0.1 * (1 - g) and o > 0.35 + 0.15 * (1 - g):
        z = max(y_lower - e_upper * (1.2 + 0.1 * (1 - g)), 0.0)
        if abs(head_yaw_rad) > pi / 18:
            yaw_scale = max(1.0 - (abs(head_yaw_rad) - pi / 18) / (pi / 4), 1.0 / 3.0)
            z *= yaw_scale
        if z < 0.1:
            z = 0.0
        else:
            dimple = (k("mouthDimpleLeft") + k("mouthDimpleRight")) * 0.5
            z = min(z * (1.0 + dimple), 1.0)
    if z > 0:
        # ne-branch (the reference's MMD ぺろっ path): sinusoidal s-curve on z.
        z = (1.0 - math.cos(z * pi)) * 0.5 if z < 0.5 else 0.5 + math.sin((z - 0.5) * pi) * 0.5

    return {
        # Vowels
        "あ": _clip(c), "い": _clip(d), "う": _clip(p), "え": _clip(m), "お": _clip(u),
        # Emotion mouth
        "にこり": _clip(nikori),
        "笑い":   _clip(R),
        "にやり": _clip(max(P, 0.0)),
        "口角上げ": _clip(mouth_corner_up),
        "困る":   _clip(T),
        "怒り":   _clip(I),
        # Tongue out
        "ぺろっ": _clip(z),
        # Surprise & look
        "びっくり": _clip(j),
        "上": _clip(look_up),
        "下": _clip(look_dn),
        # Eyes — both naming conventions; consumer picks whichever the VRM exposes
        "まばたきL": _clip(blink_L),
        "まばたきR": _clip(blink_R),
        "まばたき":   _clip(min(blink_L, blink_R)),   # both eyes closed
        "ウィンク":    _clip(blink_L),                  # = char-LEFT wink (the reference convention)
        "ウィンク右":   _clip(blink_R),                  # = char-RIGHT wink
    }


def _clip(v: float) -> float:
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return float(v)


def to_vrm_expressions(mmd_morphs: dict[str, float]) -> dict[str, float]:
    """MMD morph dict -> VRM 0.x preset weights (a/i/u/e/o/blink/joy/angry/sorrow/fun).
    Combines multiple MMD morphs into the same VRM expression with max-aggregation."""
    out: dict[str, float] = {}
    for mmd_name, w in mmd_morphs.items():
        preset = MMD_TO_VRM_PRESET.get(mmd_name)
        if preset is None:
            continue
        out[preset] = max(out.get(preset, 0.0), float(w))
    return out
