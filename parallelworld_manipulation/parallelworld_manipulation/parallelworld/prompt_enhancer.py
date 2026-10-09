"""Ground planner candidates into prompts for a single terminal-state image."""

from __future__ import annotations

import io
import json
import re
from typing import Any

import numpy as np
from PIL import Image

from .verifier import _AgentRequestBudget, _run_agent_output

PROMPT_ENHANCEMENT_VERSION = "v8"

_CONTROL_SYNTAX = re.compile(
    r"(?:"
    r"\b(?:move_to|move_pose|pi0_pick|pi0_doubled|set_gripper|"
    r"navigate_to|move_base|rotate_wrist|rotate_pitch)\b|"
    r"\b(?:xyz|target_pitch|target_yaw|tol|ori_tol|step_clip|pitch_step|"
    r"yaw_step|max_steps|max_chunks|lift_thresh|gripper_closed_thresh)\b|"
    r"\bgripper(?:\s+(?:command|setting))?\s*(?:=|:)?\s*[-+]?\d|"
    r"\[[\s,.;+\-\d]+\]|"
    r"(?<![\w])[-+]?\d+\.\d+(?![\w])|"
    r"\b\d+\s*(?:steps?|chunks?|degrees?|centimeters?|millimeters?|meters?|cm|mm)\b"
    r")",
    re.IGNORECASE,
)

_REQUIRED_FIELDS = (
    "candidate_id",
    "target_identity",
    "initial_state",
    "motion_phases",
    "contact_and_gripper",
    "object_response",
    "terminal_state",
    "candidate_contrast",
    "natural_instruction",
)
_FIELD_WORD_LIMITS = {
    "target_identity": 6,
    "initial_state": 5,
    "motion_phases": 9,
    "contact_and_gripper": 4,
    "object_response": 4,
    "terminal_state": 5,
    "candidate_contrast": 3,
    "natural_instruction": 40,
}

_MIN_NATURAL_INSTRUCTION_WORDS = 12
_STRUCTURED_LABEL = re.compile(
    r"\b(?:target identity|initial state|motion|contact(?:/gripper)?|"
    r"object response|final state|terminal state|distinction)\s*:",
    re.IGNORECASE,
)

_IDENTITY_PREFIX_WORDS = {
    "a", "an", "the", "one", "selected", "identified", "as", "intended",
    "target", "candidate", "object",
}
_POSITION_WORDS = {
    "left", "right", "leftmost", "rightmost", "upper", "lower", "top",
    "bottom", "front", "rear", "back", "middle", "center", "central",
    "nearest", "farthest", "uppermost", "lowermost", "far",
}
_RELATION_WORDS = {
    "above", "below", "behind", "beside", "between", "inside", "near",
    "on", "under",
}

_WORD_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b")
_FOLLOW_UP = re.compile(
    r"\b(?:plac(?:e|es|ed|ing)\s+(?:it|them|the|a|an|this|that|object|target)\b|"
    r"releas(?:e|es|ed|ing)|"
    r"drop(?:s|ped|ping)?|sets?\s+down|put(?:s|ting)?\s+down|"
    r"carr(?:y|ies|ied|ying)\s+(?:it|the\s+object)\s+(?:on|to|toward))\b",
    re.IGNORECASE,
)
_RELEASE_FOLLOW_UP = re.compile(
    r"\b(?:withdraw(?:s|n|ing)?|retract(?:s|ed|ing)?|"
    r"regrasp(?:s|ed|ing)?|lift(?:s|ed|ing)?\s+again|moves?\s+away)\b",
    re.IGNORECASE,
)
_ARM_TRANSLATION = re.compile(
    r"\b(?:translat(?:e|es|ed|ing)|moves?\s+(?:the\s+)?arm|"
    r"moves?\s+(?:to|toward|towards|away)|lift(?:s|ed|ing)?|"
    r"lower(?:s|ed|ing)|rais(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_ROTATION_CONFLICT = re.compile(
    r"\b(?:translat(?:e|es|ed|ing)|moves?\s+(?:to|toward|towards|away)|"
    r"lift(?:s|ed|ing)?|lower(?:s|ed|ing)|plac(?:e|es|ed|ing)|"
    r"releas(?:e|es|ed|ing)|regrasp(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_NEGATED_SIDE_CONSTRAINT = re.compile(
    r"\b(?:without|(?:does|do|will)\s+not|never)\b.*?"
    r"(?=\s*(?:[,.;]|\b(?:and\s+then|then|but|while|afterwards?)\b|$))",
    re.IGNORECASE,
)

_ROBOT_ACTORS = {"robot", "arm", "gripper", "wrist", "finger", "fingers", "end-effector"}
_ROBOT_ACTION_VERBS = {
    "approach", "approaches", "carry", "carries", "close", "closes",
    "descend", "descends", "lift", "lifts", "lower", "lowers", "move",
    "moves", "open", "opens", "perform", "performs", "press", "presses",
    "raise", "raises", "reach", "reaches", "release", "releases", "rotate",
    "rotates", "slide", "slides", "tilt", "tilts", "toggle", "toggles",
    "translate", "translates", "turn", "turns", "twist", "twists", "nudge",
    "nudges",
}
_TRANSITIVE_MOTION_VERBS = {
    "carry", "carries", "carried", "carrying", "lift", "lifts", "lifted",
    "lifting", "lower", "lowers", "lowered", "lowering", "move", "moves",
    "moved", "moving", "raise", "raises", "raised", "raising", "translate",
    "translates", "translated", "translating",
}
_WAYPOINT_ACTION_VERBS = _TRANSITIVE_MOTION_VERBS.union({
    "align", "aligns", "approach", "approaches", "descend", "descends",
    "reach", "reaches",
})
_PICK_ACTION_VERBS = _WAYPOINT_ACTION_VERBS.union({"close", "closes", "grasp", "grasps"})
_CONTACT_ACTION_VERBS = _ROBOT_ACTION_VERBS.union({"push", "pushes", "pull", "pulls"})
_GRIPPER_ACTION_VERBS = {"close", "closes", "open", "opens", "release", "releases"}
_ROTATION_ACTION_VERBS = {"rotate", "rotates", "tilt", "tilts", "turn", "turns"}
_DIRECTION_WORDS = {
    "above", "across", "along", "away", "below", "beside", "into", "near",
    "onto", "over", "through", "to", "toward", "towards", "under",
}
_PREPOSITION_WORDS = _DIRECTION_WORDS.union(
    {"at", "by", "for", "from", "of", "outside", "with", "within"}
)
_TARGET_REFERENCE_WORDS = {"it", "them", "object", "objects", "target", "targets"}
_OPEN_STATE = re.compile(r"\b(?:open|opens|opened|opening)\b", re.IGNORECASE)
_HOLDING_STATE = re.compile(
    r"\b(?:closed|grasp|grasped|grasping|held|holding|secure|secured|securely)\b",
    re.IGNORECASE,
)
_NO_CONTACT = re.compile(
    r"\b(?:no\s+contact|without\s+(?:making\s+)?(?:contact|contacting|touching)|"
    r"does\s+not\s+(?:touch|contact)|never\s+(?:touches|contacts)|"
    r"avoid(?:s|ed|ing)?\s+(?:all\s+)?contact|"
    r"(?:keep(?:s|ing)?|maintain(?:s|ed|ing)?)\s+(?:a\s+)?(?:safe\s+)?clearance|"
    r"clear\s+of\s+(?:contact|the\s+target|the\s+object))\b",
    re.IGNORECASE,
)
_STATIONARY_STATE = re.compile(
    r"\b(?:stationary|motionless|unmoved|does\s+not\s+move|stays?\s+(?:still|fixed)|"
    r"stays?\s+in\s+place|remains?\s+(?:still|fixed|stationary|unmoved|where\s+it\s+is)|"
    r"(?:is|remains?)\s+unaffected)\b",
    re.IGNORECASE,
)
_MOVING_RESPONSE = re.compile(
    r"\b(?:follow|follows|followed|following|move|moves|moved|moving|travel|travels|"
    r"transport|transports|carried|rotates?|tilts?)\b",
    re.IGNORECASE,
)
_RIGID_HELD_RESPONSE = re.compile(
    r"\b(?:follow(?:s|ed|ing)?\s+(?:the\s+)?(?:gripper|wrist)|"
    r"follow(?:s|ed|ing)?\s+rigidly|"
    r"(?:move|moves|moved|moving|travel|travels|traveled|traveling)\s+"
    r"(?:rigidly\s+)?with\s+(?:the\s+)?(?:gripper|wrist)|"
    r"(?:rotate|rotates|rotated|rotating|tilt|tilts|tilted|tilting)\s+rigidly|"
    r"carried\s+(?:rigidly\s+)?(?:by|with)\s+(?:the\s+)?gripper)\b",
    re.IGNORECASE,
)
_NONRIGID_HELD_MOTION = re.compile(
    r"\b(?:independent(?:ly)?|loose(?:ly)?|slip(?:s|ped|ping)?|"
    r"swing(?:s|ing)?|swung|wobbl(?:e|es|ed|ing)|lag(?:s|ged|ging)?|"
    r"drift(?:s|ed|ing)?|dangl(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_SETTLED_STATE = re.compile(
    r"\b(?:rest|rests|rested|resting|settle|settles|settled|settling|supported)\b",
    re.IGNORECASE,
)
_STOPPED_STATE = re.compile(
    r"\b(?:endpoint|final|holds?|maintains?|pose|remains?|stays?|stop|stops|stopped|"
    r"together|waypoint)\b",
    re.IGNORECASE,
)


def contains_control_syntax(text: str) -> bool:
    """Return whether a rendered prompt leaks simulator/API control syntax."""
    return bool(_CONTROL_SYNTAX.search(str(text))) or str(text).lstrip().startswith("{")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text)))


def _has_explicit_robot_actor(text: str, verbs: set[str]) -> bool:
    """Return whether a robot component grammatically precedes an action verb."""
    tokens = [word.lower() for word in _WORD_RE.findall(str(text))]
    for index, token in enumerate(tokens):
        if token not in _ROBOT_ACTORS:
            continue
        # Allow short modifiers such as "robot arm smoothly moves", but do not
        # accept an imperative whose only robot mention occurs after the motion.
        if any(word in verbs for word in tokens[index + 1 : index + 6]):
            return True
    return False


def _target_is_direct_motion_object(text: str, target: Any) -> bool:
    """Detect prose such as ``Move the bowl`` rather than ``move toward it``."""
    target_words = _significant_words(_safe_phrase(target)).union(
        _TARGET_REFERENCE_WORDS
    )
    if not target_words:
        return False
    tokens = [word.lower() for word in _WORD_RE.findall(str(text))]
    for index, token in enumerate(tokens):
        if token not in _TRANSITIVE_MOTION_VERBS:
            continue
        if token == "lower" and index and tokens[index - 1] in {"a", "an", "the"}:
            # Spatial adjective in a grounded identity such as "the lower bowl".
            continue
        following = tokens[index + 1 : index + 11]
        target_positions = [
            offset for offset, word in enumerate(following) if word in target_words
        ]
        if not target_positions:
            continue
        direction_positions = [
            offset for offset, word in enumerate(following) if word in _DIRECTION_WORDS
        ]
        if not direction_positions or min(target_positions) < min(direction_positions):
            return True
    return False


def _target_is_motion_subject(text: str, target: Any) -> bool:
    """Detect a stationary target acting as the mover, including pronoun aliases."""
    target_words = _significant_words(_safe_phrase(target)).union(
        _TARGET_REFERENCE_WORDS
    )
    for clause in re.split(r"[.;!?]+", str(text)):
        tokens = [word.lower() for word in _WORD_RE.findall(clause)]
        for index, token in enumerate(tokens):
            if token not in _WAYPOINT_ACTION_VERBS:
                continue
            preceding = tokens[max(0, index - 16) : index]
            offset = max(0, index - 16)
            target_positions = [
                offset + position
                for position, word in enumerate(preceding)
                if word in target_words
            ]
            if not target_positions:
                continue
            robot_positions = [
                offset + position
                for position, word in enumerate(preceding)
                if word in _ROBOT_ACTORS
            ]
            if not robot_positions:
                return True
            target_position = max(target_positions)
            robot_position = max(robot_positions)
            if robot_position < target_position:
                between = tokens[robot_position + 1 : target_position]
                if not any(word in _PREPOSITION_WORDS for word in between):
                    return True
            elif target_position < robot_position:
                between = tokens[target_position + 1 : robot_position]
                if any(word in _PREPOSITION_WORDS for word in between):
                    return True
    return False


def _has_rotation_conflict(text: str) -> bool:
    """Reject later arm motion while allowing the required phrase ``in place``."""
    normalized = re.sub(r"\bin\s+place\b", "", str(text), flags=re.IGNORECASE)
    # Negated side constraints are desired ("without translating the arm"),
    # not evidence that the candidate actually performs another action.
    normalized = _NEGATED_SIDE_CONSTRAINT.sub("", normalized)
    return bool(_ROTATION_CONFLICT.search(normalized))


def _safe_phrase(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" `\"'.")
    if not text or contains_control_syntax(text):
        return ""
    return text[:500]


def _clause(value: Any, max_words: int) -> str:
    text = _safe_phrase(value)
    text = re.sub(r"[.!?;]+", ",", text).strip(" ,:;-\t\r\n")
    return " ".join(text.split()[: max(1, int(max_words))]).strip(" ,:;-")


def _embedded_clause(value: Any) -> str:
    text = _clause(value, 1000)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


def _identity_tokens(value: Any) -> list[str]:
    words = [
        word.strip(" ,.;:()[]{}")
        for word in _clause(value, 100).split()
        if word.strip(" ,.;:()[]{}")
    ]
    while words and words[0].lower() in _IDENTITY_PREFIX_WORDS:
        words.pop(0)
    return words


def _position_tokens(words: list[str]) -> list[str]:
    positions: list[str] = []
    for word in words:
        normalized = word.lower()
        hyphen_parts = normalized.split("-")
        if normalized in _POSITION_WORDS or (
            len(hyphen_parts) == 2
            and all(part in _POSITION_WORDS for part in hyphen_parts)
        ):
            if normalized not in {item.lower() for item in positions}:
                positions.append(word)
        if len(positions) == 2:
            break
    return positions


def _horizontal_side(value: Any) -> str | None:
    parts: set[str] = set()
    for word in _WORD_RE.findall(str(value or "").lower()):
        parts.add(word)
        parts.update(word.split("-"))
    has_left = any(word == "left" or word.startswith("leftmost") for word in parts)
    has_right = any(word == "right" or word.startswith("rightmost") for word in parts)
    if has_left == has_right:
        return None
    return "left" if has_left else "right"


def _fallback_target_identity(target_value: Any, qualifier_value: Any) -> str:
    """Keep a compact object noun phrase while adding a useful location cue."""
    target_words = _identity_tokens(target_value)
    qualifier_words = _identity_tokens(qualifier_value)
    if not target_words:
        target_words = qualifier_words or ["intended", "target", "in", "front", "view"]
        qualifier_words = []

    target_positions = _position_tokens(target_words)
    qualifier_positions = _position_tokens(qualifier_words)
    if not target_positions and qualifier_positions:
        combined = qualifier_positions + target_words
    elif (
        qualifier_words
        and qualifier_words[0].lower() in _RELATION_WORDS
        and not target_positions
    ):
        combined = target_words + qualifier_words
    else:
        combined = target_words
    if len(combined) > 6:
        final_word = combined[-1]
        combined = combined[:5]
        if final_word.lower() not in {word.lower() for word in combined}:
            combined.append(final_word)
    return " ".join(combined)


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Strip runtime-only values such as cached image arrays from the LLM payload."""
    return {str(key): value for key, value in candidate.items() if not str(key).startswith("_")}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _gripper_mode(candidate: dict[str, Any]) -> str:
    primitive = str(candidate.get("primitive", ""))
    arguments = candidate.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    command = arguments.get("gripper")
    if primitive == "pi0_pick":
        return "holding"
    if primitive == "release":
        return "released"
    if isinstance(command, (int, float, np.number)):
        return "holding" if float(command) > 0 else "open"
    if primitive in {"move_to", "move_pose", "set_gripper"}:
        return "open"
    if primitive in {"rotate_wrist", "rotate_pitch"}:
        return "holding"
    visual = candidate.get("visual_action")
    state = str(visual.get("gripper_state", "")) if isinstance(visual, dict) else ""
    if any(token in state.lower() for token in ("hold", "closed", "grasp")):
        return "holding"
    if "open" in state.lower():
        return "open"
    return "unchanged"


def _action_boundary(candidate: dict[str, Any]) -> tuple[str, str]:
    primitive = str(candidate.get("primitive", ""))
    mode = _gripper_mode(candidate)
    if primitive == "pi0_pick":
        return "pick_hold", "the closed gripper keeps the target securely held at its final pose"
    if primitive in {"move_to", "move_pose", "navigate_to", "move_base"}:
        if mode == "holding":
            if primitive == "move_pose":
                return (
                    "move_holding",
                    "the held object follows the gripper's position and orientation "
                    "rigidly and stops securely at the final waypoint",
                )
            return (
                "move_holding",
                "the held object follows the gripper rigidly and stops securely at the "
                "final waypoint",
            )
        return (
            "move_open",
            "the gripper remains open at its final waypoint and supported objects stay at rest",
        )
    if primitive == "release":
        return (
            "release_settle",
            "the object remains naturally supported and the open gripper stays at its final pose",
        )
    if primitive == "set_gripper":
        if mode == "holding":
            return (
                "close_in_place",
                "the closed gripper maintains the resulting grasp while the arm stays fixed",
            )
        return (
            "open_in_place",
            "the open gripper and arm keep their resulting pose while any object settles naturally",
        )
    if primitive in {"rotate_wrist", "rotate_pitch"}:
        if mode == "holding":
            motion = "tilts" if primitive == "rotate_pitch" else "rotates"
            return (
                "rotate_holding",
                f"the held object {motion} rigidly with the wrist while arm position and "
                "grasp stay fixed",
            )
        return (
            "rotate_open",
            "the open gripper keeps its final position while the target object remains stationary",
        )
    if primitive == "pi0_doubled":
        return "contact_hold", "the robot preserves the immediate contact result at its final pose"
    return "hold_result", "the robot preserves the resulting pose, grasp, and object state"


def _render_structured(spec: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Render validated semantics as natural prose followed by a hard stop rule."""
    action = _safe_phrase(spec.get("natural_instruction")).strip()
    action = action.rstrip(".!? ") + "."
    _, endpoint_clause = _action_boundary(candidate)
    endpoint = endpoint_clause or "the robot preserves the immediate visible result"
    boundary = (
        f"At the endpoint, {endpoint}; after settling, keep the arm, gripper, all "
        "objects, front camera, and background motionless in the terminal-state image; "
        "do not begin the next task step or add extra movement."
    )
    return re.sub(r"\s+", " ", f"{action} {boundary}").strip()


def _target_reference(value: Any) -> str:
    target = _embedded_clause(value) or "intended target in the front view"
    if re.match(r"^(?:the|a|an)\b", target, flags=re.IGNORECASE):
        return target
    return f"the {target}"


def _fallback_natural_instruction(
    candidate: dict[str, Any], target_identity: str
) -> str:
    """Write fluent bounded action prose when the multimodal enhancer is unusable."""
    primitive = str(candidate.get("primitive", ""))
    mode = _gripper_mode(candidate)
    target = _target_reference(target_identity)
    visual = candidate.get("visual_action")
    visual = visual if isinstance(visual, dict) else {}
    if primitive == "pi0_pick":
        return (
            f"The robot gripper moves to {target}, closes securely around it, and lifts it "
            "clear. The target follows the gripper and ends held steadily at the "
            "reached pose."
        )
    if primitive in {"move_to", "move_pose"}:
        if mode == "holding":
            pose_detail = (
                " while changing position and wrist orientation together"
                if primitive == "move_pose"
                else ""
            )
            return (
                f"The robot gripper carries the securely held object smoothly toward {target}"
                f"{pose_detail} "
                "and stops at the intended waypoint without changing the closed grasp."
            )
        contact = _clause(visual.get("contact"), 10)
        if not re.match(
            r"^(?:stops?|avoids?|does not|never)\b", contact, flags=re.IGNORECASE
        ):
            contact = "stops at the intended waypoint without touching it"
        pose_detail = (
            " while changing position and wrist orientation together"
            if primitive == "move_pose"
            else ""
        )
        return (
            f"The open robot gripper moves smoothly toward {target}{pose_detail} and "
            f"{contact}. The "
            "gripper remains open, and the target remains stationary throughout this "
            "movement."
        )
    if primitive == "release":
        return (
            f"The robot gripper opens at its current pose and releases the held object onto "
            f"{target}. The object settles naturally on its support while the open "
            "gripper remains in place."
        )
    if primitive == "set_gripper":
        if mode == "holding":
            return (
                f"The stationary robot gripper closes securely around {target} without "
                "moving the arm. The target remains steady, and the resulting grasp "
                "stays closed at the same pose."
            )
        return (
            f"The robot gripper opens around {target} without moving the arm. Any supported "
            "object settles naturally at the same location."
        )
    if primitive in {"rotate_wrist", "rotate_pitch"}:
        motion = "tilts" if primitive == "rotate_pitch" else "rotates"
        if mode == "open":
            return (
                f"The robot wrist {motion} smoothly in place near {target} while the "
                "gripper remains open, without touching the target or translating the arm. "
                "The target remains stationary."
            )
        return (
            f"The robot wrist {motion} smoothly in place while the gripper keeps {target} "
            f"securely held. The held object {motion} rigidly with the wrist while arm "
            "position and grasp remain unchanged."
        )
    if primitive == "pi0_doubled":
        contact_motion = _clause(visual.get("motion"), 16)
        if contact_motion:
            return (
                f"The robot gripper {contact_motion}. It stops as soon as the requested "
                "immediate visible contact result is reached, without beginning another action."
            )
        return (
            f"The robot performs only the immediate contact motion on {target} and "
            "stops as soon as the requested visible result is reached."
        )
    return (
        f"The robot performs only the requested immediate motion toward {target} and "
        "stops when its visible endpoint is reached."
    )


def _fallback_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    primitive = str(candidate.get("primitive", ""))
    visual = candidate.get("visual_action")
    visual = visual if isinstance(visual, dict) else {}
    target = _fallback_target_identity(
        visual.get("target"), visual.get("target_qualifier")
    )
    initial = "gripper starts near target area"
    visual_contact = _safe_phrase(visual.get("contact"))
    mode = _gripper_mode(candidate)

    if primitive == "pi0_pick":
        phases = ["descend to target", "close around rim", "lift target clear"]
        contact = "fingers establish secure contact"
        response = "target follows gripper rigidly"
        terminal = "target remains securely held aloft"
    elif primitive in {"move_to", "move_pose"}:
        phases = [
            "arm moves with coordinated pose alignment"
            if primitive == "move_pose"
            else "arm translates smoothly to target waypoint"
        ]
        if mode == "holding":
            contact = "closed grasp remains stable"
            response = "held target follows gripper"
            terminal = "object and gripper stop together"
        else:
            contact = (
                visual_contact
                if visual_contact and _word_count(visual_contact) <= 6
                else "open gripper avoids target contact"
            )
            response = "target remains stationary"
            terminal = "gripper remains open at waypoint"
    elif primitive == "release":
        phases = ["open gripper at current pose"]
        contact = "finger contact ends locally"
        response = "object settles onto support"
        terminal = "object rests and gripper stays open"
    elif primitive == "set_gripper":
        if mode == "holding":
            phases = ["close gripper with arm fixed"]
            contact = "fingers maintain current grasp"
            response = "held object remains steady"
            terminal = "closed gripper and arm stay fixed"
        else:
            phases = ["open gripper with arm fixed"]
            contact = "fingers end current contact"
            response = "supported object settles passively"
            terminal = "open gripper and arm stay fixed"
    elif primitive in {"rotate_wrist", "rotate_pitch"}:
        phases = [
            "wrist tilts smoothly in place"
            if primitive == "rotate_pitch"
            else "wrist rotates smoothly in place"
        ]
        if mode == "holding":
            contact = "gripper preserves current grasp"
            response = (
                "held object tilts rigidly"
                if primitive == "rotate_pitch"
                else "held object rotates rigidly"
            )
            terminal = "wrist stops with grasp unchanged"
        else:
            contact = "open gripper avoids contact"
            response = "target remains stationary"
            terminal = "open wrist stops in place"
    elif primitive == "pi0_doubled":
        phases = ["perform the immediate contact motion"]
        contact = "contact changes only as required"
        response = "target responds naturally to contact"
        terminal = "immediate contact result remains visible"
    else:
        phases = ["perform the intended immediate physical motion"]
        contact = "contact changes only as required"
        response = "objects respond naturally"
        terminal = "immediate visible result is reached"

    contrast = {
        "pi0_pick": "grasp lift endpoint",
        "move_to": "direct waypoint approach",
        "move_pose": "coordinated pose approach",
        "release": "local release endpoint",
        "set_gripper": "gripper-only state change",
        "rotate_wrist": "in-place wrist rotation",
        "rotate_pitch": "in-place wrist tilt",
        "pi0_doubled": "immediate contact result",
    }.get(primitive, "target motion endpoint")
    return {
        "candidate_id": str(candidate.get("id", "candidate")),
        "target_identity": target,
        "initial_state": initial,
        "motion_phases": phases,
        "contact_and_gripper": contact,
        "object_response": response,
        "terminal_state": terminal,
        "candidate_contrast": contrast,
        "natural_instruction": _fallback_natural_instruction(candidate, target),
    }


def semantic_fallback(candidate: dict[str, Any]) -> str:
    """Build a complete, action-bounded visual prompt without an LLM."""
    return _render_structured(_fallback_spec(candidate), candidate)


def _significant_words(text: str) -> set[str]:
    ignored = {
        "the", "this", "that", "with", "from", "into", "onto", "near", "beside",
        "object", "target", "one", "identified", "lower", "upper", "right", "left",
    }
    return {
        word.lower()
        for word in _WORD_RE.findall(text)
        if len(word) >= 3 and word.lower() not in ignored
    }


def _validate_spec(
    spec: Any, candidate: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return None, ["candidate result must be an object"]
    candidate_id = str(candidate.get("id", ""))
    if str(spec.get("candidate_id", "")) != candidate_id:
        errors.append(f"candidate_id must be {candidate_id!r}")
    for field in _REQUIRED_FIELDS:
        value = spec.get(field)
        if field == "motion_phases":
            if not isinstance(value, list) or not (1 <= len(value) <= 4):
                errors.append("motion_phases must contain one to four phases")
            elif any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append("every motion phase must be a non-empty string")
        elif not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    if errors:
        return None, errors

    for field, limit in _FIELD_WORD_LIMITS.items():
        if field == "motion_phases":
            count = sum(_word_count(item) for item in spec[field])
        else:
            count = _word_count(spec[field])
        if count > limit:
            errors.append(f"{field} has {count} words; maximum is {limit}")
    if errors:
        return None, errors

    text_fields = [
        str(spec[field])
        for field in _REQUIRED_FIELDS
        if field not in {"candidate_id", "motion_phases"}
    ] + [str(item) for item in spec["motion_phases"]]
    if any(contains_control_syntax(value) for value in text_fields):
        errors.append("structured text leaks API or numeric control syntax")

    natural_instruction = str(spec["natural_instruction"]).strip()
    natural_word_count = _word_count(natural_instruction)
    if natural_word_count < _MIN_NATURAL_INSTRUCTION_WORDS:
        errors.append(
            "natural_instruction must contain at least "
            f"{_MIN_NATURAL_INSTRUCTION_WORDS} words"
        )
    if _STRUCTURED_LABEL.search(natural_instruction):
        errors.append("natural_instruction must be fluent prose without field labels")
    if "\n" in natural_instruction or re.search(r"(?:^|\s)[-*]\s", natural_instruction):
        errors.append("natural_instruction must be prose rather than a list")
    if not re.search(r"[.!?]", natural_instruction):
        errors.append("natural_instruction must contain complete sentences")

    visual = candidate.get("visual_action")
    visual = visual if isinstance(visual, dict) else {}
    expected_target = _safe_phrase(visual.get("target"))
    expected_words: set[str] = set()
    actual_words = _significant_words(str(spec["target_identity"]))
    if expected_target:
        expected_words = _significant_words(expected_target)
        if expected_words and not expected_words.intersection(actual_words):
            errors.append("target_identity is not grounded in the candidate target")
        expected_side = _horizontal_side(expected_target)
        actual_side = _horizontal_side(spec["target_identity"])
        if expected_side and actual_side != expected_side:
            errors.append(
                f"target_identity must preserve the candidate's {expected_side}-side qualifier"
            )
        natural_words = _significant_words(natural_instruction)
        if expected_words and not expected_words.intersection(natural_words):
            errors.append("natural_instruction is not grounded in the candidate target")
        natural_side = _horizontal_side(natural_instruction)
        if expected_side and natural_side != expected_side:
            errors.append(
                "natural_instruction must preserve the candidate's "
                f"{expected_side}-side qualifier"
            )

    initial_words = _significant_words(str(spec["initial_state"]))
    initial_tokens = {
        word.lower() for word in _WORD_RE.findall(str(spec["initial_state"]))
    }
    robot_words = {"robot", "arm", "gripper", "wrist", "base", "fingers"}
    if not robot_words.intersection(initial_words):
        errors.append("initial_state must locate the robot or gripper")
    target_reference_words = expected_words.union(actual_words)
    if (
        target_reference_words
        and not target_reference_words.intersection(initial_words)
        and "target" not in initial_tokens
    ):
        errors.append("initial_state must relate the robot to the candidate target")

    contact_lower = str(spec["contact_and_gripper"]).lower()
    if not re.search(r"\b(?:contact|gripper|finger|grasp|touch|rim|open|close)", contact_lower):
        errors.append("contact_and_gripper must state contact or gripper behavior")

    primitive = str(candidate.get("primitive", ""))
    mode = _gripper_mode(candidate)
    boundary_rule, _ = _action_boundary(candidate)
    contact_text = str(spec["contact_and_gripper"])
    response_text = str(spec["object_response"])
    terminal_text = str(spec["terminal_state"])
    core = " ".join(
        [str(item) for item in spec["motion_phases"]]
        + [
            str(spec["contact_and_gripper"]),
            str(spec["object_response"]),
            str(spec["terminal_state"]),
            str(spec["candidate_contrast"]),
        ]
    )
    if primitive in {"pi0_pick", "move_to", "move_pose"} and _FOLLOW_UP.search(core):
        errors.append("the candidate expands into a later placing or grasp-change action")
    if primitive == "pi0_pick" and not (
        re.search(r"\b(?:close|closes|grasp|grips?)\b", core, re.IGNORECASE)
        and re.search(r"\blift(?:s|ed|ing)?\b", core, re.IGNORECASE)
    ):
        errors.append("pick must include grasp closure and lifting")
    if primitive in {"move_to", "move_pose"} and not re.search(
        r"\b(?:move|moves|moved|moving|carry|carries|carried|carrying|"
        r"approach|approaches|align|aligns|reach|reaches|translate|translates|"
        r"translated|translating|descend|descends|descended|descending|lower|"
        r"lowers|lowered|lowering|raise|raises|raised|raising|lift|lifts|lifted|"
        r"lifting)\b",
        core,
        re.IGNORECASE,
    ):
        errors.append("move must include visible waypoint motion")
    if primitive == "release" and not re.search(
        r"\b(?:open|opens|opened|opening|release|releases|released|releasing)\b",
        core,
        re.IGNORECASE,
    ):
        errors.append("release must include local gripper opening")
    if primitive in {"move_to", "move_pose"} and _gripper_mode(candidate) == "open":
        if re.search(r"\b(?:held|holding|secure grasp|closed grasp)\b", core, re.IGNORECASE):
            errors.append("the candidate claims a held object although its gripper is open")
    if primitive == "release" and _RELEASE_FOLLOW_UP.search(core):
        errors.append("release must end in place without arm withdrawal or regrasping")
    if primitive == "set_gripper" and _ARM_TRANSLATION.search(core):
        fixed_core = re.sub(
            r"\bwithout\s+(?:moving|translating|rotating)\s+(?:the\s+)?arm\b",
            "",
            core,
            flags=re.IGNORECASE,
        )
        if _ARM_TRANSLATION.search(fixed_core):
            errors.append("set_gripper may change only the gripper while the arm remains fixed")
    if primitive in {"rotate_wrist", "rotate_pitch"} and _has_rotation_conflict(core):
        errors.append("rotation may not translate the arm or change the grasp")

    if boundary_rule == "move_open":
        if not (_OPEN_STATE.search(contact_text) and _NO_CONTACT.search(contact_text)):
            errors.append("open move contact state must say the gripper is open with no contact")
        if not _STATIONARY_STATE.search(response_text):
            errors.append("open move object_response must say the target remains stationary")
    elif boundary_rule == "move_holding":
        if not _HOLDING_STATE.search(contact_text):
            errors.append("holding move contact state must preserve the closed grasp")
        if not _RIGID_HELD_RESPONSE.search(response_text):
            errors.append(
                "holding move object_response must follow the moving gripper rigidly"
            )
        if _NONRIGID_HELD_MOTION.search(response_text):
            errors.append("holding move object_response may not move independently or loosely")
        if not _STOPPED_STATE.search(terminal_text):
            errors.append("holding move terminal_state must stop at the reached waypoint")
    elif boundary_rule == "pick_hold":
        if not _MOVING_RESPONSE.search(response_text):
            errors.append("pick object_response must follow the lifting gripper")
        if not _HOLDING_STATE.search(terminal_text):
            errors.append("pick terminal_state must keep the target securely held")
    elif boundary_rule == "release_settle":
        if not _SETTLED_STATE.search(response_text):
            errors.append("release object_response must settle on its support")
        if not _OPEN_STATE.search(terminal_text):
            errors.append("release terminal_state must leave the gripper open")
    elif boundary_rule in {"close_in_place", "open_in_place"}:
        expected_state = _HOLDING_STATE if boundary_rule == "close_in_place" else _OPEN_STATE
        if not expected_state.search(contact_text + " " + terminal_text):
            errors.append("gripper-only action must preserve its commanded final finger state")
    elif boundary_rule == "rotate_holding":
        if not _HOLDING_STATE.search(contact_text):
            errors.append("holding rotation must preserve the closed grasp")
        if not _RIGID_HELD_RESPONSE.search(response_text):
            errors.append("held object must rotate rigidly with the gripper")
        if _NONRIGID_HELD_MOTION.search(response_text):
            errors.append("held object may not slip, swing, or move independently")
    elif boundary_rule == "rotate_open":
        if not (_OPEN_STATE.search(contact_text) and _NO_CONTACT.search(contact_text)):
            errors.append("open rotation must keep the gripper open without contact")
        if not _STATIONARY_STATE.search(response_text):
            errors.append("open rotation object_response must keep the target stationary")

    prose = natural_instruction
    if boundary_rule in {"move_holding", "rotate_holding"} and (
        _NONRIGID_HELD_MOTION.search(prose)
    ):
        errors.append("natural_instruction contradicts the rigidly held object motion")
    if primitive in {"pi0_pick", "move_to", "move_pose"} and _FOLLOW_UP.search(prose):
        errors.append("natural_instruction expands into a later action")
    if primitive == "pi0_pick" and not (
        re.search(r"\b(?:close|closes|closed|closing|grasp|grips?)\b", prose, re.I)
        and re.search(r"\blift(?:s|ed|ing)?\b", prose, re.I)
    ):
        errors.append("natural_instruction must describe grasp closure and lifting")
    if primitive in {"move_to", "move_pose"} and not re.search(
        r"\b(?:move|moves|moved|moving|carry|carries|carried|carrying|"
        r"approach|approaches|align|aligns|reach|reaches|translate|translates|"
        r"translated|translating|descend|descends|descended|descending|lower|"
        r"lowers|lowered|lowering|raise|raises|raised|raising|lift|lifts|lifted|"
        r"lifting)\b",
        prose,
        re.IGNORECASE,
    ):
        errors.append("natural_instruction must describe visible waypoint motion")
    if primitive == "release" and not re.search(
        r"\b(?:open|opens|opened|opening|release|releases|released|releasing)\b",
        prose,
        re.IGNORECASE,
    ):
        errors.append("natural_instruction must describe local gripper opening")
    if primitive in {"move_to", "move_pose"} and mode == "open":
        if re.search(r"\b(?:held|holding|secure grasp|closed grasp)\b", prose, re.I):
            errors.append("natural_instruction claims a held object with an open gripper")
    if primitive == "release" and _RELEASE_FOLLOW_UP.search(prose):
        errors.append("natural_instruction continues after the local release")
    if primitive == "set_gripper" and _ARM_TRANSLATION.search(prose):
        fixed_prose = re.sub(
            r"\bwithout\s+(?:moving|translating|rotating)\s+(?:the\s+)?arm\b",
            "",
            prose,
            flags=re.IGNORECASE,
        )
        if _ARM_TRANSLATION.search(fixed_prose):
            errors.append("natural_instruction moves the arm during a gripper-only action")
    if primitive in {"rotate_wrist", "rotate_pitch"} and _has_rotation_conflict(prose):
        errors.append("natural_instruction translates the arm during an in-place rotation")

    if primitive == "pi0_pick":
        if not _has_explicit_robot_actor(prose, _PICK_ACTION_VERBS):
            errors.append("natural_instruction must name the robot gripper as the pick actor")
        if not _HOLDING_STATE.search(prose):
            errors.append("natural_instruction must end with the picked target securely held")
    elif primitive == "pi0_doubled":
        if not _has_explicit_robot_actor(prose, _CONTACT_ACTION_VERBS):
            errors.append("natural_instruction must name the robot as the contact-action actor")
        if not _STOPPED_STATE.search(prose):
            errors.append("natural_instruction must stop at the immediate contact result")
    elif primitive in {"move_to", "move_pose"}:
        if not _has_explicit_robot_actor(prose, _WAYPOINT_ACTION_VERBS):
            errors.append(
                "natural_instruction must explicitly name the moving robot arm or gripper as actor"
            )
        if boundary_rule == "move_open":
            if _target_is_direct_motion_object(prose, spec["target_identity"]):
                errors.append(
                    "natural_instruction makes the stationary target the direct object of motion"
                )
            if _target_is_motion_subject(prose, spec["target_identity"]):
                errors.append(
                    "natural_instruction makes the stationary target the moving subject"
                )
            if not _OPEN_STATE.search(prose):
                errors.append("natural_instruction must say the moving gripper remains open")
            if not _NO_CONTACT.search(prose):
                errors.append("natural_instruction must say the open move makes no contact")
            if not _STATIONARY_STATE.search(prose):
                errors.append("natural_instruction must say the target remains stationary")
        else:
            if not _HOLDING_STATE.search(prose):
                errors.append("natural_instruction must preserve the held object's closed grasp")
            if not _STOPPED_STATE.search(prose):
                errors.append("natural_instruction must stop the held object at the waypoint")
        if primitive == "move_pose" and not re.search(
            r"\b(?:align|aligns|orientation|oriented|pose|reorient|reorients|"
            r"rotate|rotates|tilt|tilts|wrist)\b",
            prose,
            re.IGNORECASE,
        ):
            errors.append("move_pose natural_instruction must describe simultaneous orientation control")
    elif primitive == "release":
        if not _has_explicit_robot_actor(prose, _GRIPPER_ACTION_VERBS):
            errors.append("natural_instruction must name the gripper as the release actor")
        if not _SETTLED_STATE.search(prose):
            errors.append("natural_instruction must let the released object settle locally")
        if not re.search(r"\b(?:in\s+place|same\s+pose|stays?|remains?)\b", prose, re.I):
            errors.append("natural_instruction must keep the arm or gripper at the release pose")
    elif primitive == "set_gripper":
        if not _has_explicit_robot_actor(prose, _GRIPPER_ACTION_VERBS):
            errors.append("natural_instruction must name the gripper or fingers as actor")
        if not re.search(
            r"\b(?:arm\s+(?:(?:remains?|stays?|is)\s+)?(?:fixed|still|stationary)|"
            r"without\s+moving\s+(?:the\s+)?arm|same\s+(?:arm\s+)?pose|"
            r"gripper\s+(?:(?:remains?|stays?|is)\s+)?(?:fixed|still|stationary))\b",
            prose,
            re.IGNORECASE,
        ):
            errors.append("natural_instruction must keep the arm fixed during a gripper-only action")
    elif primitive in {"rotate_wrist", "rotate_pitch"}:
        if not _has_explicit_robot_actor(prose, _ROTATION_ACTION_VERBS):
            errors.append("natural_instruction must name the robot wrist as the rotation actor")
        if mode == "holding":
            if not _HOLDING_STATE.search(prose):
                errors.append("natural_instruction must preserve the held object's grasp")
        else:
            if not (_OPEN_STATE.search(prose) and _NO_CONTACT.search(prose)):
                errors.append("natural_instruction must keep the rotating gripper open without contact")
            if not _STATIONARY_STATE.search(prose):
                errors.append("natural_instruction must keep the rotation target stationary")
    return (dict(spec), []) if not errors else (None, errors)


def _extract_group(
    output: Any, candidates: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    if isinstance(output, dict):
        payload = output
    else:
        text = str(output or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, ["empty or non-JSON enhancer output"]
        try:
            payload = json.loads(match.group(0))
        except Exception as exc:
            return None, [f"invalid JSON: {exc}"]
    items = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None, ["top-level candidates must be a list"]
    expected_ids = [str(candidate.get("id", "")) for candidate in candidates]
    indexed: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            return None, ["every candidate result must be an object"]
        item_id = str(item.get("candidate_id", ""))
        if item_id in indexed:
            return None, [f"duplicate candidate_id {item_id!r}"]
        indexed[item_id] = item
    if set(indexed) != set(expected_ids):
        return None, [
            "candidate IDs must match the complete input group: "
            f"expected {expected_ids!r}, received {list(indexed)!r}"
        ]
    validated: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    candidates_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    for candidate_id in expected_ids:
        spec, item_errors = _validate_spec(indexed[candidate_id], candidates_by_id[candidate_id])
        if item_errors:
            errors.extend(f"{candidate_id}: {error}" for error in item_errors)
        elif spec is not None:
            validated[candidate_id] = spec
    if not errors and len(validated) > 1:
        contrasts = [
            re.sub(r"\W+", " ", validated[candidate_id]["candidate_contrast"].lower()).strip()
            for candidate_id in expected_ids
        ]
        if len(set(contrasts)) != len(contrasts):
            errors.append("candidate_contrast must visibly distinguish every candidate")
    return (validated, []) if not errors else (None, errors)


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    try:
        return json.dumps(output, ensure_ascii=False, default=_json_default)
    except Exception:
        return str(output)


class PromptEnhancer:
    """One group-aware multimodal call, with one bounded repair attempt."""

    def __init__(
        self, *, model: str = "codex:gpt-5.6-sol", base_url: str | None = None
    ) -> None:
        from .verifier import _make_agent
        from parallelworld_manipulation.codex_transport import ENHANCER_OUTPUT_SCHEMA

        self._model = model
        self._last_records: dict[str, dict[str, Any]] = {}
        self._agent = _make_agent(
            model=model,
            base_url=base_url,
            output_schema=ENHANCER_OUTPUT_SCHEMA,
            system_prompt=(
                "You ground a complete group of LIBERO robot candidates for an "
                "terminal-state image world model. Inspect the labeled native-resolution "
                "front image belonging to every candidate, the real task instruction, "
                "and the full candidate group. Predict one front-view image showing the "
                "immediate result after the action and any natural settling. Do not "
                "request a video, frame sequence, animation, or montage. Return JSON only as "
                "{\"candidates\":[...]}. Each item must contain exactly these semantic "
                "fields: candidate_id, target_identity, initial_state, motion_phases "
                "(one to four short strings), contact_and_gripper, object_response, "
                "terminal_state, candidate_contrast, and natural_instruction. The first "
                "eight fields are internal validation data. natural_instruction supplies "
                "the action prose shown to the image model: write one or two fluent, "
                "complete sentences with 12 to 40 words and no labels or list syntax. "
                "It must name the grounded target, describe only the immediate physical "
                "action and its reached visible state, and state the exact gripper and "
                "contact behavior naturally. Do not include post-action endpoint or "
                "freeze wording because the program deterministically appends the "
                "primitive-specific boundary. Ground object identity, pose, "
                "and spatial qualifiers in the image. Make candidate_contrast describe "
                "this candidate's own visible discriminator without naming an unwanted "
                "alternative action. Primitive arguments define the exact action scope "
                "and gripper command; visual_action supplies grounded appearance and "
                "motion semantics but may not expand that scope. Describe only this "
                "immediate candidate and stop at its visible outcome: no subsequent task "
                "step, autonomous continuation, invented placement, withdrawal, regrasp, "
                "or unrelated manipulation. A release only opens at the current pose; "
                "a gripper-only command keeps the arm fixed; a wrist rotation keeps "
                "position and grasp fixed; a move preserves the commanded gripper state. "
                "Every robot motion must explicitly name the moving robot component as "
                "the grammatical actor. For an open-gripper move, the robot arm or "
                "gripper moves toward or above the target without contact, while the "
                "target object remains stationary. Never make that stationary target, "
                "the target, the object, or a pronoun such as it the subject or direct "
                "object of move, carry, translate, raise, lower, or lift. Only a "
                "pick or a move with an already closed grasp may describe an object as "
                "moving with the gripper. A pose move must also describe its simultaneous "
                "orientation change, and an already held object follows position and "
                "orientation rigidly without slipping, swinging, lagging, or moving "
                "independently. A rotation with an open gripper keeps the target "
                "stationary and must not invent a held object; a holding rotation makes "
                "the held object rotate rigidly with the wrist. "
                "Never put API names, coordinates, simulator axes, numeric commands, "
                "tolerances, budgets, or control syntax into semantic fields. Keep all "
                "fields concise; use at most 6 words for target_identity, 5 for "
                "initial_state (locating the robot/gripper relative to the target), "
                "9 total across motion_phases, 4 for "
                "contact_and_gripper, 4 for object_response, 5 for terminal_state, "
                "3 for candidate_contrast, and 40 for natural_instruction. Preserve any "
                "explicit left/right side "
                "qualifier in the candidate target; never mirror or replace it from "
                "image interpretation."
            ),
        )

    @staticmethod
    def _request_content(
        *,
        candidates: list[dict[str, Any]],
        task: str,
        fronts: dict[str, np.ndarray],
        rollout_steps: int,
        repair_errors: list[str] | None = None,
        prior_output: str | None = None,
    ) -> list[Any]:
        from pydantic_ai import BinaryContent

        # Retain the caller signature for existing rollout clients; video length
        # does not condition a single terminal-state image.
        del rollout_steps
        manifest = []
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            image = np.asarray(fronts[candidate_id], dtype=np.uint8)
            manifest.append({
                "candidate_id": candidate_id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            })
        payload: dict[str, Any] = {
            "mode": "repair" if repair_errors else "initial",
            "task": str(task),
            "prediction_target": "single_terminal_state_image",
            "candidate_ids": [str(candidate["id"]) for candidate in candidates],
            "candidates": [_public_candidate(candidate) for candidate in candidates],
            "front_image_manifest": manifest,
            "required_output_fields": list(_REQUIRED_FIELDS),
            "instruction": (
                "Return one complete structured item for every candidate ID. The images "
                "below are candidate-labeled current front observations. Describe the "
                "immediate action and its settled endpoint for one predicted image."
            ),
        }
        if repair_errors:
            payload["validation_errors_to_repair"] = repair_errors
            payload["prior_output"] = str(prior_output or "")[:12000]
        content: list[Any] = [json.dumps(payload, ensure_ascii=False, default=_json_default)]
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            image = np.asarray(fronts[candidate_id], dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(image).convert("RGB").save(buf, format="PNG")
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("._")
            content.append(f"Candidate ID: {candidate_id}; current front view.")
            content.append(BinaryContent(
                data=buf.getvalue(),
                media_type="image/png",
                identifier=f"{safe_id or 'candidate'}_start_front.png",
                vendor_metadata={"detail": "high"},
            ))
        return content

    def enhance_batch(
        self,
        *,
        candidates: list[dict[str, Any]],
        task: str,
        fronts: dict[str, np.ndarray],
        rollout_steps: int,
    ) -> dict[str, dict[str, Any]]:
        candidates = [dict(candidate) for candidate in candidates]
        if not candidates:
            return {}
        candidate_ids = [str(candidate.get("id", "")).strip() for candidate in candidates]
        if any(not candidate_id for candidate_id in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate ids must be non-empty and unique")

        missing_fronts = [candidate_id for candidate_id in candidate_ids if candidate_id not in fronts]
        if missing_fronts:
            reason = f"missing starting front image for {missing_fronts!r}"
            records = self._fallback_records(
                candidates=candidates,
                reason=reason,
                repair_attempted=False,
                raw_output=None,
                repair_output=None,
                repair_reason=None,
            )
            self._last_records = records
            return records

        initial_content = self._request_content(
            candidates=candidates,
            task=task,
            fronts=fronts,
            rollout_steps=rollout_steps,
        )
        raw_output: Any = None
        request_budget = _AgentRequestBudget()
        initial_call_failed = False
        try:
            raw_output = _run_agent_output(
                self._agent, initial_content, operation="enhancer_initial",
                budget=request_budget,
            )
            specs, errors = _extract_group(raw_output, candidates)
        except Exception as exc:
            initial_call_failed = True
            specs, errors = None, [f"enhancer call failed: {exc}"]
        if specs is not None:
            records = self._records_from_specs(
                candidates=candidates,
                specs=specs,
                source="llm",
                raw_output=_output_text(raw_output),
                repair_output=None,
                repair_reason=None,
                repair_attempted=False,
            )
            self._last_records = records
            return records

        initial_error = "; ".join(errors)
        # Repair only an actual response that failed semantic validation.
        # API failures alone have no model output to repair.
        if initial_call_failed:
            records = self._fallback_records(
                candidates=candidates,
                reason=(f"{initial_error}; request budget exhausted after {request_budget.used} attempts"
                        if not request_budget.remaining else initial_error),
                repair_attempted=False,
                raw_output=_output_text(raw_output),
                repair_output=None,
                repair_reason=None,
            )
            self._last_records = records
            return records
        repair_content = self._request_content(
            candidates=candidates,
            task=task,
            fronts=fronts,
            rollout_steps=rollout_steps,
            repair_errors=errors,
            prior_output=_output_text(raw_output),
        )
        repair_output: Any = None
        try:
            repair_output = _run_agent_output(
                self._agent, repair_content, operation="enhancer_repair",
                # Independent budget: first repair request plus three retries.
                attempts=4,
            )
            repaired_specs, repair_errors = _extract_group(repair_output, candidates)
        except Exception as exc:
            repaired_specs, repair_errors = None, [f"repair call failed: {exc}"]
        if repaired_specs is not None:
            records = self._records_from_specs(
                candidates=candidates,
                specs=repaired_specs,
                source="llm_repair",
                raw_output=_output_text(raw_output),
                repair_output=_output_text(repair_output),
                repair_reason=initial_error,
                repair_attempted=True,
            )
            self._last_records = records
            return records

        fallback_reason = (
            f"initial validation: {initial_error}; repair validation: "
            f"{'; '.join(repair_errors)}"
        )
        records = self._fallback_records(
            candidates=candidates,
            reason=fallback_reason,
            repair_attempted=True,
            raw_output=_output_text(raw_output),
            repair_output=_output_text(repair_output),
            repair_reason=initial_error,
        )
        self._last_records = records
        return records

    def _records_from_specs(
        self,
        *,
        candidates: list[dict[str, Any]],
        specs: dict[str, dict[str, Any]],
        source: str,
        raw_output: str | None,
        repair_output: str | None,
        repair_reason: str | None,
        repair_attempted: bool,
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            spec = specs[candidate_id]
            instruction = _render_structured(spec, candidate)
            if contains_control_syntax(instruction) or not (50 <= _word_count(instruction) <= 90):
                spec = _fallback_spec(candidate)
                instruction = _render_structured(spec, candidate)
                record_source = "semantic_fallback"
                fallback_reason = "rendered LLM structure violated the final prompt contract"
            else:
                record_source = source
                fallback_reason = None
            boundary_rule, _ = _action_boundary(candidate)
            records[candidate_id] = {
                "prompt_enhancement_version": PROMPT_ENHANCEMENT_VERSION,
                "candidate_id": candidate_id,
                "instruction": instruction,
                "source": record_source,
                "fallback_reason": fallback_reason,
                "repair_reason": repair_reason,
                "repair_attempted": bool(repair_attempted),
                "raw_output": raw_output,
                "repair_output": repair_output,
                "structured": spec,
                "model": getattr(self, "_model", "unknown"),
                "word_count": _word_count(instruction),
                "action_boundary_rule": boundary_rule,
            }
        return records

    def _fallback_records(
        self,
        *,
        candidates: list[dict[str, Any]],
        reason: str,
        repair_attempted: bool,
        raw_output: str | None,
        repair_output: str | None,
        repair_reason: str | None,
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            spec = _fallback_spec(candidate)
            instruction = _render_structured(spec, candidate)
            boundary_rule, _ = _action_boundary(candidate)
            records[candidate_id] = {
                "prompt_enhancement_version": PROMPT_ENHANCEMENT_VERSION,
                "candidate_id": candidate_id,
                "instruction": instruction,
                "source": "semantic_fallback",
                "fallback_reason": str(reason),
                "repair_reason": repair_reason,
                "repair_attempted": bool(repair_attempted),
                "raw_output": raw_output,
                "repair_output": repair_output,
                "structured": spec,
                "model": getattr(self, "_model", "unknown"),
                "word_count": _word_count(instruction),
                "action_boundary_rule": boundary_rule,
            }
        return records

    def enhance(
        self,
        *,
        candidate: dict[str, Any],
        task: str = "",
        front: np.ndarray | None = None,
        rollout_steps: int = 49,
    ) -> str:
        """Backward-compatible single-candidate wrapper around grouped enhancement."""
        item = dict(candidate)
        candidate_id = str(item.get("id") or "candidate")
        item["id"] = candidate_id
        if front is None:
            front = np.zeros((1, 1, 3), dtype=np.uint8)
        return self.enhance_batch(
            candidates=[item],
            task=task,
            fronts={candidate_id: np.asarray(front, dtype=np.uint8)},
            rollout_steps=rollout_steps,
        )[candidate_id]["instruction"]
