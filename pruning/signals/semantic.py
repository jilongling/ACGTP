"""Semantic and attention relevance signals."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/scores/semantic_anchors.py
# ---------------------------------------------------------------------------
import re
from typing import Any, Dict, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour / object / spatial-relation term dictionaries
# ─────────────────────────────────────────────────────────────────────────────

COLOR_WORDS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "beige", "navy", "cyan", "magenta",
    "dark", "light", "transparent",
}

# Spatial relation prepositions / conjunctions that signal spatial meaning
RELATION_PREPOSITIONS = {
    "next_to", "beside", "near", "close_to",
    "on", "onto", "top_of", "on_top_of",
    "in", "inside", "within", "in_front_of", "behind",
    "left_of", "right_of",
    "between", "under", "below", "above", "over",
    "far_from", "away_from",
    "adjacent_to", "opposite", "across_from",
    "after", "before",  # temporal — deprioritised but not excluded
}

# Object nouns frequently found in robot manipulation instructions
OBJECT_NOUNS = {
    "bowl", "cup", "mug", "plate", "dish", "pan", "pot", "spatula",
    "knife", "fork", "spoon", "chopstick",
    "bottle", "can", "box", "container", "tray", "rack",
    "shelf", "table", "counter", "sink", "stove", "burner",
    "drawer", "cabinet", "lid", "cover", "sponge", "cloth",
    "apple", "banana", "egg", "cheese", "bread",
    "toy", "block", "ball", "cube", "cylinder",
    "lego", "shape", "item", "object", "target", "source",
    "ramekin", "sugar_bowl", "cutting_board", "cuttingboard",
}

SPATIAL_PREPOSITIONS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in RELATION_PREPOSITIONS) + r")\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stable-sort / set utilities (shared across selector and audit)
# ─────────────────────────────────────────────────────────────────────────────

def _stable_topk(scores: np.ndarray, k: int, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Stable top-k: sort descending by score, then ascending by index (lexsort).

    Parameters
    ----------
    scores : np.ndarray
        [N] scores (higher = better).
    k : int
        Maximum number of indices to return.
    valid : np.ndarray or None
        [N] boolean mask. Invalid tokens are excluded.

    Returns
    -------
    np.ndarray
        [min(k, M)] indices sorted by (score desc, index asc), where M is the
        number of valid tokens.
    """
    if valid is None:
        valid = np.ones(scores.shape, dtype=bool)
    adj = np.where(valid, scores, -np.inf)
    neg_adj = -adj
    order = np.lexsort((np.arange(scores.shape[0]), neg_adj))
    return order[:k]


def jaccard(set_a, set_b) -> float:
    """Jaccard similarity between two collections (lists, sets, or arrays)."""
    a = set(int(x) for x in set_a)
    b = set(int(x) for x in set_b)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Instruction parser
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    # Split on underscores AND on standard word boundaries
    # e.g. "pick_up_the_black_bowl" -> ['pick', 'up', 'the', 'black', 'bowl']
    parts = text.replace("_", " ").lower()
    return re.findall(r"[a-z]+", parts)


ACTION_VERBS = {
    "pick", "up", "place", "put", "set", "drop", "push", "pull",
    "open", "close", "shut", "turn", "twist", "rotate",
    "slide", "move", "carry", "hold", "grasp", "release",
    "take", "get", "put", "swap", "replace", "lift", "lower",
    "hang", "fold", "unfold", "wipe", "clean", "wash",
    "pour", "scoop", "scoop_up", "pick_up", "put_down",
    "close_the", "open_the", "pick_it_up", "put_it_down",
}

STOP_WORDS_ACTION = {
    "pick", "up", "put", "place", "set", "down", "into",
    "to", "on", "in", "and", "or", "a", "an", "it", "the",
    "this", "that", "with", "from", "onto",
}


def parse_instruction_terms(instruction: str) -> Dict[str, Any]:
    """Extract structured scene-layout terms from a task instruction string.

    This is purely syntactic — no visual model involved. The instruction is
    typically derived from task_name in the environment (e.g.
    "pick_up_the_black_bowl_between_the_plate_and_the_red_mug").

    Output schema (all keys always present):
      target:               Dict with keys: object (str), attributes (list[str])
      references:           List[Dict] with keys: object (str), attributes (list[str])
      relations:            List[Dict] with keys: type (str), target (str), references (list[str])
      actions:              List[str] of action verbs
      raw_parse:            Dict with parsed raw fields for debugging
      instruction_is_meaningful: bool

    Guarantees:
      - action verbs do NOT appear in reference object list
      - relation prepositions do NOT appear in reference object list
      - target object is NOT in the reference list
    """
    if not instruction or not isinstance(instruction, str):
        return _empty_parse_result()

    raw_lower = instruction.lower()
    tokens = _tokenise(instruction)

    color_tokens = [t for t in tokens if t in COLOR_WORDS]

    relation_terms = list(RELATION_PREPOSITIONS.intersection(
        w.lower() for w in re.findall(r"[a-z_]+", raw_lower)
    ))

    object_tokens = [t for t in tokens if t in OBJECT_NOUNS]

    # ── Extract action verbs (before removing them from reference list) ─────────
    actions: list[str] = []
    action_candidates = set(tokens) & ACTION_VERBS
    if action_candidates:
        for t in tokens:
            if t in ACTION_VERBS and t not in STOP_WORDS_ACTION:
                if t not in actions:
                    actions.append(t)
    # Heuristic: "pick_up_the_X" → action = "pick up"
    raw_action_phrases: list[str] = []
    raw_lower_tokens = re.findall(r"[a-z_]+", raw_lower)
    for t in raw_lower_tokens:
        if t in {"pick_up", "put_down", "put", "place", "pick"}:
            if t not in raw_action_phrases:
                raw_action_phrases.append(t)
        elif t in {"open", "close", "push", "pull", "slide"}:
            if t not in raw_action_phrases:
                raw_action_phrases.append(t)

    # ── 1. Detect "between X and Y" → reference objects ────────────────────
    between_refs: list[str] = []
    try:
        bi = tokens.index("between")
        ref1_parts, ref2_parts = [], []
        phase = 0
        for t in tokens[bi + 1:]:
            if t == "and":
                phase = 1
                continue
            if phase == 0:
                ref1_parts.append(t)
            else:
                ref2_parts.append(t)
        between_refs = ref1_parts + ref2_parts
    except ValueError:
        pass

    # ── 2. Detect "<prep> <X>" → reference objects ─────────────────────────
    # Skip preps, relation words, and action verbs
    exclusion_set = (STOP_WORDS_ACTION | set(RELATION_PREPOSITIONS) | ACTION_VERBS |
                     {"the", "a", "an", "it", "this", "that"})
    prep_refs: list[str] = []
    for i, t in enumerate(tokens):
        matched_prep = None
        for prep in RELATION_PREPOSITIONS:
            prep_tokens = prep.split("_")
            prep_len = len(prep_tokens)
            if prep_len >= 2:
                if i + prep_len <= len(tokens) and tokens[i:i + prep_len] == prep_tokens:
                    matched_prep = prep
                    break
            else:
                if tokens[i] == prep:
                    matched_prep = prep
                    break
        if matched_prep is not None:
            prep_len = len(matched_prep.split("_"))
            if i + prep_len < len(tokens):
                nxt = tokens[i + prep_len]
                if nxt not in {"the", "a", "an"}:
                    prep_refs.append(nxt)
                elif i + prep_len + 1 < len(tokens):
                    prep_refs.append(tokens[i + prep_len + 1])

    # ── Build raw reference list (no action verbs, no relation preps) ───────
    raw_refs: list[str] = []
    for ref in between_refs + prep_refs:
        if ref in exclusion_set:
            continue
        raw_refs.append(ref)

    # Deduplicate preserving order
    seen = set()
    ref_terms: list[str] = []
    for x in raw_refs:
        if x not in seen and not seen.add(x):
            ref_terms.append(x)

    # ── 3. Target: first colour+object phrase, distinct from references ────────
    target_object: str = ""
    target_attrs: list[str] = []
    excluded_refs = set(ref_terms)

    if color_tokens:
        first_color_idx = tokens.index(color_tokens[0])
        target_attrs = [color_tokens[0]]
        if first_color_idx + 1 < len(tokens):
            nxt = tokens[first_color_idx + 1]
            if nxt not in STOP_WORDS_ACTION and len(nxt) > 2 and nxt not in excluded_refs:
                target_object = nxt
            else:
                target_object = color_tokens[0]
        else:
            target_object = color_tokens[0]
    elif object_tokens:
        for t in object_tokens:
            if t not in excluded_refs and t not in STOP_WORDS_ACTION:
                target_object = t
                break
    else:
        meaningful = [t for t in tokens if t not in STOP_WORDS_ACTION and len(t) > 2]
        if meaningful:
            target_object = meaningful[0]

    # ── Build structured relation list ───────────────────────────────────────
    relations: list[Dict[str, Any]] = []
    for rel_type in sorted(relation_terms, key=len, reverse=True):
        rel_type_tokens = rel_type.split("_")
        rel_type_len = len(rel_type_tokens)
        for i in range(len(tokens)):
            if tokens[i:i + rel_type_len] == rel_type_tokens:
                if i + rel_type_len < len(tokens):
                    nxt = tokens[i + rel_type_len]
                    if nxt in {"the", "a", "an"} and i + rel_type_len + 1 < len(tokens):
                        nxt = tokens[i + rel_type_len + 1]
                    rel_refs = [nxt] if nxt not in STOP_WORDS_ACTION else []
                else:
                    rel_refs = []
                relations.append({
                    "type": rel_type,
                    "target": target_object,
                    "references": rel_refs,
                })
                break

    # ── Build structured output ───────────────────────────────────────────────
    result = {
        "target": {
            "object": target_object,
            "attributes": [a for a in target_attrs if a],
        },
        "references": [
            {"object": r, "attributes": []} for r in ref_terms
        ],
        "relations": relations,
        "actions": list(dict.fromkeys(a for a in raw_action_phrases if a)),
        "raw_parse": {
            "parsed_target_terms": [target_object] + [a for a in target_attrs if a],
            "parsed_reference_terms": ref_terms,
            "parsed_relation_terms": relation_terms,
            "instruction_tokens": tokens,
        },
        "has_color": bool(color_tokens),
        "has_object": bool(object_tokens),
        "has_relation": bool(relation_terms),
        "instruction_is_meaningful": bool(color_tokens or object_tokens or relation_terms),
    }
    return result


def _empty_parse_result() -> Dict[str, Any]:
    """Return an empty parse result with all required keys."""
    return {
        "target": {"object": "", "attributes": []},
        "references": [],
        "relations": [],
        "actions": [],
        "raw_parse": {
            "parsed_target_terms": [],
            "parsed_reference_terms": [],
            "parsed_relation_terms": [],
            "instruction_tokens": [],
        },
        "has_color": False,
        "has_object": False,
        "has_relation": False,
        "instruction_is_meaningful": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Weak fallback scoring from geometry signals (NOT true semantic scores)
# ─────────────────────────────────────────────────────────────────────────────

def _weak_fallback_scores(
    n: int,
    scene_result: Optional[Dict[str, Any]],
    component_scores: Optional[np.ndarray],
    boundary_scores: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build very low-confidence fallback scores from geometry signals.

    These are NOT semantic — they merely replicate what scene_layout already
    captures. semantic_confidence will be set to 0.0 to signal that no real
    visual model was used.

    Returns:
        Dict with keys: semantic_target_scores, semantic_reference_scores,
        semantic_relation_scores, semantic_goal_scores, semantic_anchor_scores
        (each [N] float array).
    """
    # Zero arrays by default — geometry-only fallback is NOT semantic
    zero = np.zeros(n, dtype=np.float32)
    return {
        "semantic_target_scores": zero,
        "semantic_reference_scores": zero,
        "semantic_relation_scores": zero,
        "semantic_goal_scores": zero,
        "semantic_anchor_scores": zero,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_task_semantic_anchors(
    instruction: Optional[str],
    rgb: Optional[np.ndarray],
    token_u: Optional[np.ndarray],
    token_v: Optional[np.ndarray],
    token_depth: Optional[np.ndarray],
    scene_result: Optional[Dict[str, Any]],
    config: Optional[Any] = None,
    *,
    semantic_backend: str = "none",
    semantic_enabled: bool = False,
    w_semantic_target: float = 1.0,
    w_semantic_reference: float = 0.7,
    w_semantic_relation: float = 0.5,
    w_semantic_goal: float = 0.9,
    target_cap_ratio: float = 0.25,
    reference_cap_ratio: float = 0.20,
    relation_cap_ratio: float = 0.15,
    hard_ratio: float = 0.20,
    release_quota_when_unavailable: bool = True,
    grid_h: int = 16,
    grid_w: int = 16,
) -> Dict[str, Any]:
    """Compute task-semantic anchor scores for token protection.

    This is the main entry point. When ``semantic_enabled=False`` or
    ``semantic_backend="none"``, the function returns a low-confidence fallback
    result (semantic_unavailable=True, semantic_confidence=0.0).

    Parameters
    ----------
    instruction : str or None
        Natural-language task instruction (e.g. from task_name).
    rgb : array or None
        RGB frame [H, W, 3]. Currently unused — placeholder for future
        open-vocab detector integration.
    token_u, token_v : array or None
        [N] integer pixel grid coordinates per token.
    token_depth : array or None
        [N] depth values per token (metres).
    scene_result : dict or None
        Return value of ``compute_scene_layout_scores``. Used only for
        weak fallback scoring.
    config : object or None
        Resolved PruningHookConfig (unused in this version; reserved for future use).
    semantic_backend : str
        One of {"none", "grounding_dino", "owl_vit", "lseg"}.
        Currently only "none" is implemented; others raise NotImplementedError.
    semantic_enabled : bool
        Master switch. When False the function returns fallback immediately.
    w_semantic_target / _reference / _relation / _goal : float
        Per-category weight for semantic anchor score mixture.
    target_cap_ratio / reference_cap_ratio / relation_cap_ratio : float
        Maximum fraction of total tokens that can be assigned to each category.
    hard_ratio : float
        Fraction of keep_k reserved as hard-protect for the semantic branch.
    release_quota_when_unavailable : bool
        If True and semantic_unavailable=True, the semantic quota is NOT
        pre-allocated (available for other branches to use).
    grid_h, grid_w : int
        Token grid dimensions.

    Returns
    -------
    dict
        Keys:
          semantic_target_scores:     [N] float scores [0,1] for target object tokens.
          semantic_reference_scores:  [N] float scores [0,1] for reference object tokens.
          semantic_relation_scores:   [N] float scores [0,1] for spatial-relation region tokens.
          semantic_goal_scores:       [N] float scores [0,1] for goal/container region tokens.
          semantic_anchor_scores:     [N] float scores [0,1] = max of above four.
          semantic_unavailable:       bool (True = no real visual detector used).
          semantic_confidence:        float in [0,1] (0.0 for fallback, >0.8 for real detector).
          semantic_fallback_reason:    str or None.
          semantic_backend_used:       str (the backend string).
          semantic_enabled_by_config: bool (the enabled flag from config).
          release_quota:              bool (True when semantic unavailable and release_quota=True).
          parsed_target_terms:        list[str].
          parsed_reference_terms:      list[str].
          parsed_relation_terms:       list[str].
          parsed_instruction_is_meaningful: bool.
          instruction_tokens:         list[str].
          target_token_count:         int.
          reference_token_count:      int.
          relation_region_token_count: int.
          goal_token_count:           int.
          semantic_anchor_token_count: int.
          target_cap_k:               int.
          reference_cap_k:            int.
          relation_cap_k:             int.
          hard_semantic_quota:        int (fraction of keep_k for semantic hard protect).
          w_target: float, w_reference: float, w_relation: float, w_goal: float.
          target_cap_ratio: float, reference_cap_ratio: float, relation_cap_ratio: float.
    """
    # Determine n from available arrays
    n = 256  # default
    for arr in (token_u, token_v, token_depth):
        if arr is not None:
            a = np.asarray(arr, dtype=object).reshape(-1)
            if a.size > 0:
                n = max(n, int(a.size))

    # ── Parse instruction (supports both old flat and new nested schema) ──
    parsed = parse_instruction_terms(instruction or "")
    # Extract flat lists from either schema
    raw = parsed.get("raw_parse", {})
    if raw:  # new nested schema
        target_obj = parsed.get("target", {}).get("object", "")
        target_attrs = parsed.get("target", {}).get("attributes", [])
        ref_objs = [r.get("object", "") for r in parsed.get("references", [])]
        rel_types = [rel.get("type", "") for rel in parsed.get("relations", [])]
        raw_target_terms = raw.get("parsed_target_terms", [])
        raw_ref_terms = raw.get("parsed_reference_terms", [])
        raw_rel_terms = raw.get("parsed_relation_terms", [])
        instr_tokens = raw.get("instruction_tokens", [])
        is_meaningful = parsed.get("instruction_is_meaningful", False)
    else:  # old flat schema (backward compat)
        target_obj = ""
        target_attrs = []
        ref_objs = []
        rel_types = []
        raw_target_terms = parsed.get("parsed_target_terms", [])
        raw_ref_terms = parsed.get("parsed_reference_terms", [])
        raw_rel_terms = parsed.get("parsed_relation_terms", [])
        instr_tokens = parsed.get("instruction_tokens", [])
        is_meaningful = parsed.get("instruction_is_meaningful", False)

    result: Dict[str, Any] = {
        # Scores — zeros when semantic unavailable
        "semantic_target_scores": np.zeros(n, dtype=np.float32),
        "semantic_reference_scores": np.zeros(n, dtype=np.float32),
        "semantic_relation_scores": np.zeros(n, dtype=np.float32),
        "semantic_goal_scores": np.zeros(n, dtype=np.float32),
        "semantic_anchor_scores": np.zeros(n, dtype=np.float32),
        # Scene-layout branch scores (Task 4)
        "scene_layout_scores": np.zeros(n, dtype=np.float32),
        "target_mask": np.zeros(n, dtype=np.float32),
        "reference_mask": np.zeros(n, dtype=np.float32),
        "relation_mask": np.zeros(n, dtype=np.float32),
        "layout_anchor_mask": np.zeros(n, dtype=np.float32),
        # State flags
        "semantic_unavailable": True,
        "semantic_confidence": 0.0,
        "semantic_fallback_reason": None,
        "semantic_backend_used": str(semantic_backend),
        "semantic_enabled_by_config": bool(semantic_enabled),
        "release_quota": False,
        # Parsed instruction (flat schema, backward compat)
        "parsed_target_terms": raw_target_terms,
        "parsed_reference_terms": raw_ref_terms,
        "parsed_relation_terms": raw_rel_terms,
        "parsed_instruction_is_meaningful": is_meaningful,
        "instruction_tokens": instr_tokens,
        # Parsed instruction (new nested schema)
        "parsed_target_object": target_obj,
        "parsed_target_attributes": target_attrs,
        "parsed_reference_objects": ref_objs,
        "parsed_relation_types": rel_types,
        "parsed_actions": parsed.get("actions", []),
        "parsed_relations": parsed.get("relations", []),
        # Token counts
        "target_token_count": 0,
        "reference_token_count": 0,
        "relation_region_token_count": 0,
        "goal_token_count": 0,
        "semantic_anchor_token_count": 0,
        "scene_layout_token_count": 0,
        "target_mask_count": 0,
        "reference_mask_count": 0,
        "relation_mask_count": 0,
        "layout_anchor_mask_count": 0,
        # Caps
        "target_cap_k": 0,
        "reference_cap_k": 0,
        "relation_cap_k": 0,
        "hard_semantic_quota": 0,
        # Weights
        "w_target": float(w_semantic_target),
        "w_reference": float(w_semantic_reference),
        "w_relation": float(w_semantic_relation),
        "w_goal": float(w_semantic_goal),
        "target_cap_ratio": float(target_cap_ratio),
        "reference_cap_ratio": float(reference_cap_ratio),
        "relation_cap_ratio": float(relation_cap_ratio),
        # Scene-layout branch
        "scene_layout_branch_active": False,
        "scene_layout_branch_quota": 0,
        "scene_layout_confidence": 0.0,
        "scene_layout_available": False,
        # Raw parsed dict (for debug)
        "_parsed_full": parsed,
    }

    # ── Early exit when disabled ──────────────────────────────────────────
    if not semantic_enabled or semantic_backend == "none":
        reason = (
            "semantic_enabled=False" if not semantic_enabled
            else f"semantic_backend={semantic_backend} not implemented"
        )
        result["semantic_fallback_reason"] = reason
        result["release_quota"] = bool(release_quota_when_unavailable)
        return result

    # ── Parser-only: instruction available but no visual detector ───────────
    if semantic_backend == "parser_only":
        result["semantic_fallback_reason"] = "parser_only_backend_no_visual_detector"
        result["release_quota"] = bool(release_quota_when_unavailable)
        return result

    # ── Mock backend: synthetic masks for selector testing ───────────────────
    if semantic_backend == "mock":
        from experiments.robot.libero.acgtp_semantic_backend import get_semantic_backend
        try:
            backend = get_semantic_backend(
                backend="mock",
                grid_h=grid_h,
                grid_w=grid_w,
                seed=0,
                w_target=w_semantic_target,
                w_reference=w_semantic_reference,
                w_relation=w_semantic_relation,
                w_layout=0.6,
            )
            sl_result = backend.run(instruction=instruction, parsed=parsed)
            result["semantic_available"] = sl_result.semantic_available
            result["semantic_confidence"] = sl_result.confidence
            result["target_mask"] = sl_result.target_mask
            result["reference_mask"] = sl_result.reference_mask
            result["relation_mask"] = sl_result.relation_mask
            result["layout_anchor_mask"] = sl_result.layout_anchor_mask
            result["token_scores"] = sl_result.token_scores
            # Map to semantic score arrays (for backward compat with existing selector)
            result["semantic_target_scores"] = sl_result.target_mask
            result["semantic_reference_scores"] = sl_result.reference_mask
            result["semantic_relation_scores"] = sl_result.relation_mask
            result["semantic_goal_scores"] = sl_result.layout_anchor_mask * 0.9
            result["semantic_anchor_scores"] = sl_result.token_scores
            result["scene_layout_scores"] = sl_result.token_scores
            result["semantic_unavailable"] = not sl_result.semantic_available
            result["release_quota"] = False
            result["semantic_fallback_reason"] = None
            # Token counts
            result["target_token_count"] = int(np.sum(sl_result.target_mask > 0.5))
            result["reference_token_count"] = int(np.sum(sl_result.reference_mask > 0.5))
            result["relation_region_token_count"] = int(np.sum(sl_result.relation_mask > 0.5))
            result["goal_token_count"] = int(np.sum(sl_result.layout_anchor_mask > 0.5))
            result["semantic_anchor_token_count"] = int(np.sum(sl_result.token_scores > 0))
            result["scene_layout_token_count"] = int(np.sum(sl_result.token_scores > 0))
            result["target_mask_count"] = result["target_token_count"]
            result["reference_mask_count"] = result["reference_token_count"]
            result["relation_mask_count"] = result["relation_region_token_count"]
            result["layout_anchor_mask_count"] = result["goal_token_count"]
            # Caps
            result["target_cap_k"] = max(1, int(round(n * target_cap_ratio)))
            result["reference_cap_k"] = max(1, int(round(n * reference_cap_ratio)))
            result["relation_cap_k"] = max(1, int(round(n * relation_cap_ratio)))
            result["hard_semantic_quota"] = max(1, int(round(n * hard_ratio)))
            # Scene-layout branch active
            result["scene_layout_branch_active"] = True
            result["scene_layout_branch_quota"] = result["hard_semantic_quota"]
            result["scene_layout_confidence"] = sl_result.confidence
            result["scene_layout_available"] = sl_result.semantic_available
            result["_backend_debug"] = sl_result.debug
        except Exception as exc:
            result["semantic_fallback_reason"] = f"mock_backend_error:{exc}"
            result["release_quota"] = bool(release_quota_when_unavailable)
        return result

    # ── Real visual detector path (grounding_dino / owl_vit / lseg) ───────
    # These raise NotImplementedError until real integration is done.
    if semantic_backend in ("grounding_dino", "owl_vit", "lseg"):
        raise NotImplementedError(
            f"{semantic_backend} backend not yet wired. "
            "Set acgtp_v2_semantic_backend='none' for fallback mode, "
            "or 'mock' for selector testing."
        )

    # Unknown backend — safe fallback
    result["semantic_fallback_reason"] = f"unknown_backend:{semantic_backend}"
    result["release_quota"] = bool(release_quota_when_unavailable)
    return result


def compute_scene_layout_selected_counts_v2(
    keep_indices: np.ndarray,
    scene_result: Dict[str, Any],
    n_total: int,
) -> Dict[str, Any]:
    """Extended selection attribution that disambiguates Relation vs Residual Fill.

    Replaces the misleading "relation" label in the original function with:
      - residual_fill: tokens that are scene-relevant but NOT in support_plane /
        object_component / boundary / semantic categories.

    Compatible with the original compute_scene_layout_selected_counts signature —
    the original fields are included for backward compatibility, but comments
    clarify that "relation" is a geometry-only leftover, NOT a semantic relation.

    Args:
        keep_indices: [K] array of selected token indices.
        scene_result: Return value of ``compute_scene_layout_scores``.
        n_total: Total number of tokens.

    Returns dict includes all original fields PLUS:
      acgtp_scene_selected_residual_fill_count: int (tokens selected as geometry-only leftover).
      acgtp_scene_residual_fill_token_count: int (total geometry-only leftover tokens in grid).
      acgtp_scene_residual_fill_token_count_computed: bool.
    """
    # Delegate to original function for backward compatibility
    from .spatial import compute_scene_layout_selected_counts

    base = compute_scene_layout_selected_counts(keep_indices, scene_result, n_total)

    # Add the corrected naming
    if base.get("acgtp_scene_selected_relation_count") is not None:
        base["acgtp_scene_selected_residual_fill_count"] = base["acgtp_scene_selected_relation_count"]
    if base.get("acgtp_scene_relation_token_count") is not None:
        base["acgtp_scene_residual_fill_token_count"] = base["acgtp_scene_relation_token_count"]
    base["acgtp_scene_residual_fill_token_count_computed"] = base.get(
        "acgtp_scene_relation_token_count_computed", False
    )

    return base

# ---------------------------------------------------------------------------
# Source: pruning/scores/attention_relevance.py
# ---------------------------------------------------------------------------
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Unified result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttentionRelevanceResult:
    """Unified return type for all attention backends.

    Attributes
    ----------
    available : bool
        True only when real attention scores were loaded (precomputed with
        matching shape). False for "none" and failed "precomputed" backends.
    source : str
        One of {"none", "mock", "precomputed"}.
    confidence : float
        In [0, 1]. 0.0 = unavailable; >0.8 = real precomputed attention.
    text_to_vision_score : np.ndarray
        [N] float, text-token → vision-token attention scores (normalized).
    prefill_score : np.ndarray
        [N] float, prefill-phase decoder attention scores (normalized).
    action_to_vision_score : np.ndarray
        [N] float, action-token → vision-token attention scores (normalized).
    historical_action_score : np.ndarray
        [N] float, previous-step action attention scores (normalized).
    task_relevance_score : np.ndarray
        [N] float, composite = max(text_to_vision, prefill, action_to_vision, hist).
        Used only for candidate generation.
    task_relevance_mask : np.ndarray
        [N] bool, tokens above top-k percentile of task_relevance_score.
    debug : dict
        Per-source counts, thresholds, disabled reasons.
    """

    available: bool
    source: str
    confidence: float
    text_to_vision_score: np.ndarray
    prefill_score: np.ndarray
    action_to_vision_score: np.ndarray
    historical_action_score: np.ndarray
    task_relevance_score: np.ndarray
    task_relevance_mask: np.ndarray
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (arrays converted to lists for JSON safety)."""
        def _arr_safe(a):
            if a is None:
                return None
            try:
                return [float(x) for x in a]
            except Exception:
                return None

        d = {k: v for k, v in self.__dict__.items() if k != "debug"}
        d["debug"] = {k: v for k, v in self.debug.items()}
        d["text_to_vision_score"] = _arr_safe(self.text_to_vision_score)
        d["prefill_score"] = _arr_safe(self.prefill_score)
        d["action_to_vision_score"] = _arr_safe(self.action_to_vision_score)
        d["historical_action_score"] = _arr_safe(self.historical_action_score)
        d["task_relevance_score"] = _arr_safe(self.task_relevance_score)
        d["task_relevance_mask"] = _arr_safe(self.task_relevance_mask)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper (shared with audit)
# ─────────────────────────────────────────────────────────────────────────────

def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two boolean arrays of same shape."""
    if a.shape != b.shape:
        return 0.0
    a_b = np.asarray(a, dtype=bool)
    b_b = np.asarray(b, dtype=bool)
    inter = np.sum(a_b & b_b)
    union = np.sum(a_b | b_b)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


# ─────────────────────────────────────────────────────────────────────────────
# Backend: "none"
# ─────────────────────────────────────────────────────────────────────────────

def _backend_none(n: int) -> AttentionRelevanceResult:
    """Immediate unavailable result — zero scores, zero confidence."""
    zero = np.zeros(n, dtype=np.float32)
    return AttentionRelevanceResult(
        available=False,
        source="none",
        confidence=0.0,
        text_to_vision_score=zero,
        prefill_score=zero,
        action_to_vision_score=zero,
        historical_action_score=zero,
        task_relevance_score=zero,
        task_relevance_mask=np.zeros(n, dtype=bool),
        debug={
            "backend": "none",
            "disabled_reason": "attention_backend=none",
            "note": "attention has zero effect on selector",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backend: "mock" — synthetic scores for accounting / alignment testing
# ─────────────────────────────────────────────────────────────────────────────

def _backend_mock(
    n: int,
    grid_h: int,
    grid_w: int,
    seed: int,
    mode: str = "balanced",
    rng: Optional[np.random.Generator] = None,
) -> AttentionRelevanceResult:
    """Generate synthetic attention scores for selector stress-testing.

    Parameters
    ----------
    n : int
        Number of tokens (expected 256 for 16×16 grid).
    grid_h, grid_w : int
        Grid dimensions.
    seed : int
        RNG seed for reproducible synthetic patterns.
    mode : str
        One of {"balanced", "high_attn_geo_low", "all_background", "high_geo_attn_low",
                "partial_overlap"}. Controls the synthetic pattern.
    rng : np.random.Generator or None
        Pre-seeded RNG; if None, a new one is created from seed.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    assert n == grid_h * grid_w, f"Expected {grid_h * grid_w} tokens, got {n}"

    gripper_r, gripper_c = grid_h // 2, grid_w // 2

    # Base: uniform random scores for each attention source
    t2v = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    prefill = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    a2v = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    hist = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # Normalise each source to [0, 1] independently
    def _norm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        out = (x - lo) / (hi - lo)
        return out.astype(np.float32)

    t2v = _norm(t2v)
    prefill = _norm(prefill)
    a2v = _norm(a2v)
    hist = _norm(hist)

    # Composite = max across sources
    task_score = np.maximum.reduce([t2v, prefill, a2v, hist]).astype(np.float32)

    # Mode-specific modifications
    if mode == "balanced":
        pass  # keep uniform

    elif mode == "high_attn_geo_low":
        # Simulate VLA-IAP case: attention is high everywhere, geometry is weak.
        # Boost attention for random 30% of tokens.
        boost = rng.choice(n, size=max(1, int(n * 0.30)), replace=False)
        t2v[boost] = rng.uniform(0.7, 1.0, size=len(boost)).astype(np.float32)
        a2v[boost] = rng.uniform(0.7, 1.0, size=len(boost)).astype(np.float32)
        task_score = np.maximum.reduce([t2v, prefill, a2v, hist]).astype(np.float32)
        task_score = _norm(task_score)

    elif mode == "all_background":
        # Attention high on corners/edges (background), geometry low everywhere.
        # Use HARD masking: corners get boosted scores, everything else is suppressed
        # to near-zero. This avoids percentile threshold ambiguity where mid-range
        # tokens accidentally enter the mask due to distribution stretching.
        corners = [0, grid_w - 1, (grid_h - 1) * grid_w, (grid_h - 1) * grid_w + grid_w - 1]
        corner_r = [c // grid_w for c in corners]
        corner_c = [c % grid_w for c in corners]
        # Suppress all non-corner tokens so they never enter the top-k
        t2v[:] = 0.0
        a2v[:] = 0.0
        hist[:] = 0.0
        prefill[:] = 0.0
        for r, c in zip(corner_r, corner_c):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid_h and 0 <= cc < grid_w:
                        idx = rr * grid_w + cc
                        t2v[idx] = rng.uniform(0.7, 1.0)
                        a2v[idx] = rng.uniform(0.6, 0.9)
                        hist[idx] = rng.uniform(0.3, 0.7)
                        prefill[idx] = rng.uniform(0.2, 0.5)
        task_score = np.maximum.reduce([t2v, prefill, a2v, hist]).astype(np.float32)
        # No _norm: hard scores already in [0, 1]

    elif mode == "high_geo_attn_low":
        # Simulate geometry-dominant case: geometry branches are strong,
        # but attention is weak everywhere.
        # Weaken attention sources.
        t2v *= 0.15
        a2v *= 0.10
        hist *= 0.12
        prefill *= 0.20
        task_score = np.maximum.reduce([t2v, prefill, a2v, hist]).astype(np.float32)

    elif mode == "partial_overlap":
        # Simulate partial alignment: attention overlaps with some geometry regions.
        # Build 3 clusters around gripper.
        cluster_centers = [
            (gripper_r, gripper_c),
            (gripper_r - 2, gripper_c - 1),
            (gripper_r + 1, gripper_c + 2),
        ]
        for cr, cc in cluster_centers:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc_ = cr + dr, cc + dc
                    if 0 <= rr < grid_h and 0 <= cc_ < grid_w:
                        idx = rr * grid_w + cc_
                        t2v[idx] = rng.uniform(0.6, 1.0)
                        a2v[idx] = rng.uniform(0.5, 0.9)
        task_score = np.maximum.reduce([t2v, prefill, a2v, hist]).astype(np.float32)
        task_score = _norm(task_score)

    # Top-k mask: tokens above 70th percentile = attention high
    threshold = float(np.percentile(task_score[task_score > 0], 70)) if np.any(task_score > 0) else 0.0
    attn_mask = (task_score >= threshold).astype(bool)

    debug = {
        "backend": "mock",
        "mode": mode,
        "seed": seed,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "top_attention_threshold": threshold,
        "top_attention_count": int(np.sum(attn_mask)),
        "attn_score_mean": float(np.mean(task_score)),
        "attn_score_max": float(np.max(task_score)),
        "t2v_score_mean": float(np.mean(t2v)),
        "a2v_score_mean": float(np.mean(a2v)),
        "prefill_score_mean": float(np.mean(prefill)),
        "hist_score_mean": float(np.mean(hist)),
    }

    return AttentionRelevanceResult(
        available=True,
        source="mock",
        confidence=0.5,  # mock = intermediate confidence
        text_to_vision_score=t2v,
        prefill_score=prefill,
        action_to_vision_score=a2v,
        historical_action_score=hist,
        task_relevance_score=task_score,
        task_relevance_mask=attn_mask,
        debug=debug,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backend: "precomputed" — load from file
# ─────────────────────────────────────────────────────────────────────────────

def _backend_precomputed(
    n: int,
    grid_h: int,
    grid_w: int,
    step_index: int,
    search_dir: Optional[Path] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
) -> AttentionRelevanceResult:
    """Load precomputed attention scores from disk.

    Search order (first match wins):
      1. {search_dir}/attention_step{step_index:04d}.npy  (shape [N, 4] or [N])
      2. {search_dir}/attention_step{step_index:04d}.npz
      3. {search_dir}/attention_scores.npy (per-step data keyed by step index)
      4. {search_dir}/attention.json (list of per-step dicts)
      5. {search_dir}/attention.csv  (columns: step, token_idx, t2v, prefill, a2v, hist)

    If file is missing or shape mismatch, returns unavailable result.

    Parameters
    ----------
    n : int
        Expected number of tokens.
    grid_h, grid_w : int
        Grid dimensions.
    step_index : int
        Timestep for which to load precomputed scores.
    search_dir : Path or None
        Directory to search. Defaults to outputs/attention/ in repo root.
    token_u, token_v : np.ndarray or None
        If provided, used to verify spatial alignment.
    """
    if search_dir is None:
        search_dir = Path("/infini-data/openvla/outputs/attention")
    else:
        search_dir = Path(search_dir)

    zero = np.zeros(n, dtype=np.float32)
    disabled_reason = None

    def _fail(reason: str) -> AttentionRelevanceResult:
        return AttentionRelevanceResult(
            available=False,
            source="precomputed",
            confidence=0.0,
            text_to_vision_score=zero.copy(),
            prefill_score=zero.copy(),
            action_to_vision_score=zero.copy(),
            historical_action_score=zero.copy(),
            task_relevance_score=zero.copy(),
            task_relevance_mask=np.zeros(n, dtype=bool),
            debug={
                "backend": "precomputed",
                "disabled_reason": reason,
                "search_dir": str(search_dir),
                "step_index": step_index,
            },
        )

    # Try step-specific .npy
    step_file = search_dir / f"attention_step{step_index:04d}.npy"
    if step_file.exists():
        try:
            data = np.load(str(step_file))
            # Shape: [N] or [N, 4]
            if data.ndim == 1:
                arr = np.asarray(data, dtype=np.float32).reshape(-1)
                if arr.shape[0] != n:
                    return _fail(f"shape mismatch: expected {n}, got {arr.shape[0]}")
                t2v = arr.copy()
                prefill = arr.copy()
                a2v = arr.copy()
                hist = arr.copy()
            elif data.ndim == 2 and data.shape[1] == 4:
                if data.shape[0] != n:
                    return _fail(f"shape mismatch: expected {n}x4, got {data.shape[0]}x4")
                t2v = data[:, 0].astype(np.float32)
                prefill = data[:, 1].astype(np.float32)
                a2v = data[:, 2].astype(np.float32)
                hist = data[:, 3].astype(np.float32)
            else:
                return _fail(f"unexpected ndim/shape: {data.shape}")
        except Exception as exc:
            return _fail(f"failed to load .npy: {exc}")
    else:
        # Try .npz
        npz_file = search_dir / f"attention_step{step_index:04d}.npz"
        if npz_file.exists():
            try:
                with np.load(str(npz_file)) as npz:
                    keys = list(npz.keys())
                    t2v = npz.get("text_to_vision", npz.get("t2v", npz.get(keys[0] if keys else "")))
                    prefill = npz.get("prefill", npz.get(keys[1] if len(keys) > 1 else ""))
                    a2v = npz.get("action_to_vision", npz.get("a2v", npz.get(keys[2] if len(keys) > 2 else "")))
                    hist = npz.get("historical_action", npz.get("hist", npz.get(keys[3] if len(keys) > 3 else "")))
                    for arr_name, arr in [("t2v", t2v), ("prefill", prefill), ("a2v", a2v), ("hist", hist)]:
                        if arr is None or arr.shape[0] != n:
                            return _fail(f"{arr_name} missing or shape mismatch in npz")
                    t2v = t2v.astype(np.float32)
                    prefill = prefill.astype(np.float32)
                    a2v = a2v.astype(np.float32)
                    hist = hist.astype(np.float32)
            except Exception as exc:
                return _fail(f"failed to load .npz: {exc}")
        else:
            # Try combined .npy
            combined_file = search_dir / "attention_scores.npy"
            if combined_file.exists():
                try:
                    all_data = np.load(str(combined_file), allow_pickle=True)
                    if isinstance(all_data, np.ndarray) and all_data.ndim == 2:
                        if step_index >= all_data.shape[0]:
                            return _fail(f"step_index {step_index} out of range for {all_data.shape[0]} steps")
                        row = all_data[step_index]
                        if len(row) == 4:
                            t2v, prefill, a2v, hist = [np.asarray(x, dtype=np.float32) for x in row]
                            for arr_name, arr in [("t2v", t2v), ("prefill", prefill), ("a2v", a2v), ("hist", hist)]:
                                if arr.shape[0] != n:
                                    return _fail(f"{arr_name} shape mismatch: expected {n}, got {arr.shape[0]}")
                        else:
                            return _fail(f"row length {len(row)} != 4")
                    else:
                        return _fail(f"unexpected combined array shape: {all_data.shape}")
                except Exception as exc:
                    return _fail(f"failed to load combined .npy: {exc}")
            else:
                # Try .json
                json_file = search_dir / "attention.json"
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            all_steps = json.load(f)
                        if not isinstance(all_steps, list) or step_index >= len(all_steps):
                            return _fail(f"step_index {step_index} out of range")
                        step_data = all_steps[step_index]
                        def _json_arr(key):
                            arr = step_data.get(key, step_data.get(key.replace("_", ""), []))
                            if isinstance(arr, list):
                                a = np.array(arr, dtype=np.float32)
                            else:
                                a = np.asarray(arr, dtype=np.float32)
                            return a.reshape(-1)
                        t2v = _json_arr("text_to_vision")
                        prefill = _json_arr("prefill")
                        a2v = _json_arr("action_to_vision")
                        hist = _json_arr("historical_action")
                        for arr_name, arr in [("t2v", t2v), ("prefill", prefill), ("a2v", a2v), ("hist", hist)]:
                            if arr.shape[0] != n:
                                return _fail(f"{arr_name} shape mismatch: expected {n}, got {arr.shape[0]}")
                    except Exception as exc:
                        return _fail(f"failed to load .json: {exc}")
                else:
                    # Try .csv
                    csv_file = search_dir / "attention.csv"
                    if csv_file.exists():
                        try:
                            import csv as csv_lib
                            rows_dict: Dict[int, Dict[str, float]] = {}
                            with open(csv_file) as f:
                                reader = csv_lib.DictReader(f)
                                for row in reader:
                                    step_i = int(row.get("step", -1))
                                    tok_i = int(row.get("token_idx", -1))
                                    if step_i != step_index or tok_i < 0 or tok_i >= n:
                                        continue
                                    if step_i not in rows_dict:
                                        rows_dict[step_i] = {}
                                    rows_dict[step_i][tok_i] = {
                                        "t2v": float(row.get("t2v", 0)),
                                        "prefill": float(row.get("prefill", 0)),
                                        "a2v": float(row.get("a2v", 0)),
                                        "hist": float(row.get("hist", 0)),
                                    }
                            if step_index not in rows_dict:
                                return _fail(f"no data for step_index={step_index} in CSV")
                            rows = rows_dict[step_index]
                            if len(rows) != n:
                                return _fail(f"expected {n} tokens for step={step_index}, got {len(rows)}")
                            t2v = np.array([rows[i].get("t2v", 0) for i in range(n)], dtype=np.float32)
                            prefill = np.array([rows[i].get("prefill", 0) for i in range(n)], dtype=np.float32)
                            a2v = np.array([rows[i].get("a2v", 0) for i in range(n)], dtype=np.float32)
                            hist = np.array([rows[i].get("hist", 0) for i in range(n)], dtype=np.float32)
                        except Exception as exc:
                            return _fail(f"failed to load .csv: {exc}")
                    else:
                        return _fail(
                            f"no attention file found for step {step_index} in {search_dir}"
                        )

    # All sources loaded successfully — normalise
    def _norm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    t2v_n = _norm(t2v)
    prefill_n = _norm(prefill)
    a2v_n = _norm(a2v)
    hist_n = _norm(hist)
    task_score = np.maximum.reduce([t2v_n, prefill_n, a2v_n, hist_n]).astype(np.float32)

    # Threshold: 70th percentile of non-zero scores
    nonzero = task_score[task_score > 0]
    if nonzero.size > 0:
        threshold = float(np.percentile(nonzero, 70))
    else:
        threshold = 0.0
    attn_mask = (task_score >= threshold).astype(bool)

    return AttentionRelevanceResult(
        available=True,
        source="precomputed",
        confidence=0.9,
        text_to_vision_score=t2v_n,
        prefill_score=prefill_n,
        action_to_vision_score=a2v_n,
        historical_action_score=hist_n,
        task_relevance_score=task_score,
        task_relevance_mask=attn_mask,
        debug={
            "backend": "precomputed",
            "step_index": step_index,
            "search_dir": str(search_dir),
            "top_attention_threshold": threshold,
            "top_attention_count": int(np.sum(attn_mask)),
            "attn_score_mean": float(np.mean(task_score)),
            "attn_score_max": float(np.max(task_score)),
            "t2v_score_mean": float(np.mean(t2v_n)),
            "a2v_score_mean": float(np.mean(a2v_n)),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_relevance(
    backend: str = "none",
    n: int = 256,
    grid_h: int = 16,
    grid_w: int = 16,
    step_index: int = 0,
    search_dir: Optional[Path] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    seed: int = 0,
    mode: str = "balanced",
    rng: Optional[np.random.Generator] = None,
) -> AttentionRelevanceResult:
    """Factory function dispatching to the appropriate backend.

    Parameters
    ----------
    backend : str
        One of {"none", "mock", "precomputed"}.
    n : int
        Number of tokens.
    grid_h, grid_w : int
        Grid dimensions (n should equal grid_h * grid_w).
    step_index : int
        Timestep for precomputed file lookup.
    search_dir : Path or None
        Directory for precomputed files.
    token_u, token_v : np.ndarray or None
        Pixel coordinates (unused by backends, reserved for future alignment).
    seed : int
        RNG seed for mock backend.
    mode : str
        Synthetic pattern mode for mock backend.
    rng : np.random.Generator or None
        Pre-seeded RNG for mock backend.

    Returns
    -------
    AttentionRelevanceResult
        Always valid; if backend is unavailable, available=False and all
        scores are zero.
    """
    backend = str(backend).strip().lower()

    if backend == "none":
        return _backend_none(n)

    if backend == "mock":
        return _backend_mock(n=n, grid_h=grid_h, grid_w=grid_w, seed=seed, mode=mode, rng=rng)

    if backend == "precomputed":
        return _backend_precomputed(
            n=n, grid_h=grid_h, grid_w=grid_w,
            step_index=step_index, search_dir=search_dir,
            token_u=token_u, token_v=token_v,
        )

    # Unknown backend — treat as none
    zero = np.zeros(n, dtype=np.float32)
    return AttentionRelevanceResult(
        available=False,
        source=backend,
        confidence=0.0,
        text_to_vision_score=zero,
        prefill_score=zero,
        action_to_vision_score=zero,
        historical_action_score=zero,
        task_relevance_score=zero,
        task_relevance_mask=np.zeros(n, dtype=bool),
        debug={
            "backend": backend,
            "disabled_reason": f"unknown_backend:{backend}",
            "note": "treated as unavailable",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alignment diagnostic helpers (used by audit / selector integration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_geometry_alignment(
    attn_result: AttentionRelevanceResult,
    scene_scores: Optional[np.ndarray],
    depth_scores: Optional[np.ndarray],
    contact_scores: Optional[np.ndarray],
    motion_scores: Optional[np.ndarray],
    scene_threshold_pct: float = 70.0,
    depth_threshold_pct: float = 70.0,
    contact_threshold_pct: float = 70.0,
    motion_threshold_pct: float = 70.0,
) -> Dict[str, Any]:
    """Compute IoU between top-attention tokens and each geometry branch mask.

    Returns a dict of IoU scores and token counts for the attention_alignment_audit
    report.

    Parameters
    ----------
    attn_result : AttentionRelevanceResult
        Output of get_attention_relevance().
    scene_scores : [N] or None
    depth_scores : [N] or None
    contact_scores : [N] or None
    motion_scores : [N] or None
        Geometry branch score arrays (raw, not normalised).
    *_threshold_pct : float
        Percentile for binarising each geometry branch.

    Returns
    -------
    dict with keys:
        attention_available, attention_source, attention_confidence,
        top_attention_count,
        attention_scene_iou, attention_depth_iou,
        attention_contact_iou, attention_motion_iou,
        attention_acgtp_keep_iou (always 0 — used for audit reporting only),
        attention_only_token_count, geometry_only_token_count,
        attention_geometry_overlap_count,
        attention_background_risk_count (tokens high-attention but geometry-weak),
        attention_selected_by_final_count (always 0 — audit only),
        attention_high_low_geometry_count (synonym for attention_only_token_count),
        geometry_high_low_attention_count (synonym for geometry_only_token_count),
    """
    n = attn_result.task_relevance_score.shape[0]
    attn_mask = attn_result.task_relevance_mask

    def _threshold_mask(scores, pct):
        if scores is None:
            return np.zeros(n, dtype=bool)
        a = np.asarray(scores, dtype=np.float32).reshape(-1)
        if a.shape[0] != n:
            return np.zeros(n, dtype=bool)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros(n, dtype=bool)
        t = float(np.percentile(finite, pct))
        return (a >= t).astype(bool)

    scene_mask = _threshold_mask(scene_scores, scene_threshold_pct)
    depth_mask = _threshold_mask(depth_scores, depth_threshold_pct)
    contact_mask = _threshold_mask(contact_scores, contact_threshold_pct)
    motion_mask = _threshold_mask(motion_scores, motion_threshold_pct)

    # Combined geometry mask (any branch high)
    geom_mask = scene_mask | depth_mask | contact_mask | motion_mask

    # IoUs
    attn_scene_iou = _iou_bool(attn_mask, scene_mask)
    attn_depth_iou = _iou_bool(attn_mask, depth_mask)
    attn_contact_iou = _iou_bool(attn_mask, contact_mask)
    attn_motion_iou = _iou_bool(attn_mask, motion_mask)

    # Tokens: high attention + no geometry = attention_only
    attention_only = attn_mask & ~geom_mask
    # Tokens: high geometry + no attention = geometry_only
    geometry_only = geom_mask & ~attn_mask
    # Tokens: both high
    attn_geom_overlap = attn_mask & geom_mask
    # Tokens: high attention but ALL geometry branches weak
    attn_bg_risk = attn_mask & ~geom_mask

    return {
        "attention_available": attn_result.available,
        "attention_source": attn_result.source,
        "attention_confidence": attn_result.confidence,
        "top_attention_count": int(np.sum(attn_mask)),
        "attention_scene_iou": round(attn_scene_iou, 4),
        "attention_depth_iou": round(attn_depth_iou, 4),
        "attention_contact_iou": round(attn_contact_iou, 4),
        "attention_motion_iou": round(attn_motion_iou, 4),
        "attention_acgtp_keep_iou": 0.0,  # audit placeholder; real value from selector
        "attention_only_token_count": int(np.sum(attention_only)),
        "geometry_only_token_count": int(np.sum(geometry_only)),
        "attention_geometry_overlap_count": int(np.sum(attn_geom_overlap)),
        "attention_background_risk_count": int(np.sum(attn_bg_risk)),
        "attention_selected_by_final_count": 0,  # filled by audit after selector
        "high_attention_low_geometry_count": int(np.sum(attention_only)),
        "high_geometry_low_attention_count": int(np.sum(geometry_only)),
    }


def compute_safe_drop_diagnostic(
    attn_result: AttentionRelevanceResult,
    scene_scores: Optional[np.ndarray],
    depth_scores: Optional[np.ndarray],
    contact_scores: Optional[np.ndarray],
    motion_scores: Optional[np.ndarray],
    keep_indices: Optional[np.ndarray] = None,
    threshold_pct: Optional[float] = None,
    threshold_fixed: float = 0.50,
) -> Dict[str, Any]:
    """VLA-Cache style safe-drop diagnostic.

    safe_drop_candidate = low_scene AND low_depth AND low_contact
                        AND low_motion AND low_attention

    Reports counts of tokens that were safe to drop and how many were
    actually dropped (vs high-attention / high-geometry tokens that were
    dropped — the latter is a warning signal).

    Parameters
    ----------
    attn_result : AttentionRelevanceResult
    scene_scores, depth_scores, contact_scores, motion_scores : [N] or None
    keep_indices : np.ndarray or None
        Final selected token indices (used to compute dropped sets).
    threshold_pct : float or None
        Percentile for binarising geometry branches. When None (default),
        threshold_fixed is used instead. Use percentile mode when geometry
        scores have a smooth/uniform distribution. Use fixed threshold when
        scores are bimodal (clearly separated LOW/HIGH populations, as in
        synthetic test scores where LOW is [0,0.15] and HIGH is [0.70,1.0]).
    threshold_fixed : float
        Fixed score threshold when threshold_pct is None.

    Returns
    -------
    dict with keys:
        safe_drop_candidate_count,
        dropped_safe_candidate_count,
        dropped_high_attention_count (WARNING if high),
        dropped_high_geometry_count,
        high_attention_low_geometry_count,
        high_geometry_low_attention_count,
        safe_drop_ratio,
        warning (str or None),
    """
    n = attn_result.task_relevance_score.shape[0]

    def _threshold_mask(scores, pct_or_fixed):
        if scores is None:
            return np.zeros(n, dtype=bool)
        a = np.asarray(scores, dtype=np.float32).reshape(-1)
        if a.shape[0] != n:
            return np.zeros(n, dtype=bool)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros(n, dtype=bool)
        if pct_or_fixed is not None:
            t = float(np.percentile(finite, pct_or_fixed))
        else:
            t = float(threshold_fixed)
        return (a >= t).astype(bool)

    scene_mask = _threshold_mask(scene_scores, threshold_pct)
    depth_mask = _threshold_mask(depth_scores, threshold_pct)
    contact_mask = _threshold_mask(contact_scores, threshold_pct)
    motion_mask = _threshold_mask(motion_scores, threshold_pct)
    attn_mask = attn_result.task_relevance_mask

    # Safe-drop candidate: low on ALL branches
    low_scene = ~scene_mask
    low_depth = ~depth_mask
    low_contact = ~contact_mask
    low_motion = ~motion_mask
    low_attn = ~attn_mask

    safe_drop = low_scene & low_depth & low_contact & low_motion & low_attn

    # Token sets
    all_selected = set(int(i) for i in keep_indices) if keep_indices is not None else set()
    total_range = set(range(n))

    safe_drop_count = int(np.sum(safe_drop))

    # Among tokens that were NOT selected (dropped), how many were safe-drop candidates
    dropped_set = total_range - all_selected

    dropped_safe = sum(1 for i in dropped_set if safe_drop[i])
    dropped_high_attn = sum(1 for i in dropped_set if attn_mask[i])
    dropped_high_geo = sum(1 for i in dropped_set if (scene_mask[i] or depth_mask[i] or contact_mask[i] or motion_mask[i]))

    # Attention-only / geometry-only across the whole grid
    geom_mask = scene_mask | depth_mask | contact_mask | motion_mask
    high_attn_low_geo = int(np.sum(attn_mask & ~geom_mask))
    high_geo_low_attn = int(np.sum(geom_mask & ~attn_mask))

    # Warnings
    warning: Optional[str] = None
    total_dropped = len(dropped_set)
    if total_dropped > 0:
        high_attn_drop_ratio = dropped_high_attn / total_dropped
        if high_attn_drop_ratio > 0.30:
            warning = (
                f"POSSIBLE_ATTENTION_BACKGROUND_BIAS: "
                f"{dropped_high_attn}/{total_dropped} ({high_attn_drop_ratio:.1%}) "
                f"dropped tokens have high attention but low geometry."
            )
        elif high_geo_low_attn > high_attn_low_geo * 2:
            warning = (
                f"GEOMETRY_DOMINANT_PROTECTION: "
                f"{high_geo_low_attn} geometry-high/attention-low tokens vs "
                f"{high_attn_low_geo} attention-high/geometry-low tokens. "
                f"ACGTP geometry branches protecting action-constrained tokens that "
                f"attention ignores — this is the intended论文亮点."
            )

    safe_drop_ratio = safe_drop_count / n if n > 0 else 0.0

    return {
        "safe_drop_candidate_count": safe_drop_count,
        "dropped_safe_candidate_count": dropped_safe,
        "dropped_high_attention_count": dropped_high_attn,
        "dropped_high_geometry_count": dropped_high_geo,
        "high_attention_low_geometry_count": high_attn_low_geo,
        "high_geometry_low_attention_count": high_geo_low_attn,
        "safe_drop_ratio": round(safe_drop_ratio, 4),
        "warning": warning,
        "total_dropped": total_dropped,
        "total_tokens": n,
    }
