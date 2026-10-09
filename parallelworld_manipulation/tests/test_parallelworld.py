import argparse
import io
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robots.libero import _add_cli_args, _parallelworld_runtime
from robots.libero.minimax_h3_world_model_server import MiniMaxH3WorldModelFacade
from robots.libero.parallelworld import PARALLELWORLD_TOOL_SPEC
from robots.libero.toolkit import LiberoToolkit
from robots.libero.wan_world_model_server import (
    WAN_NEGATIVE_PROMPT,
    WanLiberoWorldModelFacade,
)
from parallelworld_manipulation.parallelworld.client import VideoWorldModelClient
from parallelworld_manipulation.parallelworld.explorer import ParallelWorldExplorer
from parallelworld_manipulation.parallelworld.prompt_enhancer import (
    PROMPT_ENHANCEMENT_VERSION,
    PromptEnhancer,
    contains_control_syntax,
    semantic_fallback,
)
from parallelworld_manipulation.parallelworld.verifier import VerifierAgent
from parallelworld_manipulation.tools.toolkit import Toolkit, ToolResult


class FakeClient:
    def __init__(self):
        self.calls = []

    def rollout(self, **kwargs):
        self.calls.append(kwargs)
        samples = kwargs["samples"]
        horizon = kwargs["horizon"]
        front = np.repeat(
            kwargs["front"][None, None], samples * horizon, axis=0
        ).reshape(samples, horizon, *kwargs["front"].shape)
        return {"front": front}


def primitives():
    obs = {
        "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    return SimpleNamespace(_last_obs=obs)


def candidate(candidate_id="open", primitive="release", arguments=None):
    return {
        "id": candidate_id,
        "primitive": primitive,
        "arguments": arguments or {},
    }


def visual_candidate(candidate_id, primitive, *, gripper_state="unchanged", arguments=None):
    target = "lower patterned bowl beside the cookie package"
    motions = {
        "pi0_pick": "descends, closes around the rim, and lifts the bowl",
        "move_to": "moves the held bowl smoothly toward the destination",
        "release": "opens over the target and releases the held bowl",
        "set_gripper": (
            "closes around the currently held object"
            if (arguments or {}).get("gripper", 0) > 0
            else "opens at the current arm pose"
        ),
    }
    outcomes = {
        "pi0_pick": "the bowl is suspended securely in the closed gripper",
        "move_to": "the held bowl and gripper stop together at the destination",
        "release": "the bowl rests on the support while the gripper stays open",
        "set_gripper": (
            "the current grasp remains securely closed"
            if (arguments or {}).get("gripper", 0) > 0
            else "the open gripper remains at the same arm pose"
        ),
    }
    return {
        "id": candidate_id,
        "primitive": primitive,
        "arguments": arguments or {},
        "rationale": f"test the {candidate_id} route",
        "visual_action": {
            "subject": "robot gripper",
            "motion": motions[primitive],
            "target": target,
            "target_qualifier": "the lower one, not the upper-right bowl",
            "spatial_relation": "directly above",
            "gripper_state": gripper_state,
            "contact": "contact remains stable only for this immediate action",
            "intended_outcome": outcomes[primitive],
        },
    }


def structured_enhancement(candidate_id, primitive):
    fields = {
        "pi0_pick": {
            "motion_phases": [
                "descend to target",
                "close around rim",
                "lift bowl clear",
            ],
            "contact_and_gripper": "fingers secure opposite rim",
            "object_response": "bowl follows gripper rigidly",
            "terminal_state": "bowl remains securely suspended",
        },
        "move_to": {
            "motion_phases": ["carry held bowl smoothly to destination"],
            "contact_and_gripper": "closed grasp remains steady",
            "object_response": "held bowl follows rigidly",
            "terminal_state": "bowl and gripper stop together",
        },
        "release": {
            "motion_phases": ["open gripper and release bowl"],
            "contact_and_gripper": "fingers open over support",
            "object_response": "released bowl settles naturally",
            "terminal_state": "bowl rests gripper stays open",
        },
        "set_gripper_close": {
            "motion_phases": ["close gripper with arm fixed"],
            "contact_and_gripper": "fingers maintain current grasp",
            "object_response": "held object remains steady",
            "terminal_state": "gripper closed arm remains fixed",
        },
        "set_gripper_open": {
            "motion_phases": ["open gripper with arm fixed"],
            "contact_and_gripper": "fingers end current contact",
            "object_response": "supported object settles passively",
            "terminal_state": "gripper open arm remains fixed",
        },
    }[primitive]
    natural_instructions = {
        "pi0_pick": (
            "The gripper descends to the lower patterned bowl, closes around its rim, "
            "and lifts it clear. The bowl ends securely held above the surface."
        ),
        "move_to": (
            "The robot carries the lower patterned bowl smoothly to the destination "
            "while keeping a secure grasp. The bowl and gripper stop together."
        ),
        "release": (
            "The gripper opens over the support and releases the lower patterned bowl. "
            "The bowl settles naturally while the open gripper stays at the same pose."
        ),
        "set_gripper_close": (
            "The gripper closes securely around the lower patterned bowl without moving "
            "the arm. The resulting grasp remains closed at the same pose."
        ),
        "set_gripper_open": (
            "The gripper opens around the lower patterned bowl without moving the arm. "
            "The supported bowl settles naturally while the arm stays fixed."
        ),
    }
    return {
        "candidate_id": candidate_id,
        "target_identity": "lower patterned bowl beside cookie package",
        "initial_state": "gripper starts above bowl",
        **fields,
        "candidate_contrast": f"distinct {candidate_id} endpoint",
        "natural_instruction": natural_instructions[primitive],
    }


def assert_rendered_prompt_contract(prompt):
    words = re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", prompt)
    sentences = [
        part for part in re.split(r"(?<=[.!?])\s+", prompt.strip()) if part
    ]
    assert 50 <= len(words) <= 90
    assert 2 <= len(sentences) <= 3
    assert "terminal-state image" in prompt.lower()
    assert "final frame" not in prompt.lower()
    assert "do not begin the next task step" in prompt.lower()
    assert "extra movement" in prompt.lower()
    assert "motionless" in prompt.lower()
    assert not re.search(
        r"\b(?:target identity|initial state|motion|contact/gripper|"
        r"object response|final state|distinction)\s*:",
        prompt,
        re.IGNORECASE,
    )
    assert not contains_control_syntax(prompt)


def test_tts_switch_controls_natural_language_rollout_sampling():
    for enabled, expected in ((False, 1), (True, 3)):
        client = FakeClient()
        explorer = ParallelWorldExplorer(
            primitives=primitives(), client=client, tts=enabled,
            tts_samples=3, rollout_steps=2,
        )
        result = explorer.decide(candidates=[candidate()])

        assert client.calls[0]["samples"] == expected
        assert client.calls[0]["stochastic"] is enabled
        assert client.calls[0]["horizon"] == 2
        assert "instruction" in client.calls[0]
        assert result["tts"]["enabled"] is enabled
        assert result["_image_cam_bytes"].startswith(b"\x89PNG")
        assert result["_image_wrist_bytes"] is None


def test_parallelworld_uses_calibration_rgb_and_marks_projected_endpoint():
    height = width = 64
    calibration_front = np.zeros((height, width, 3), dtype=np.uint8)
    calibration_front[:, : width // 2] = [10, 20, 30]
    calibration_wrist = np.full((height, width, 3), 7, dtype=np.uint8)

    class Env:
        def raw_obs(self):
            # Robosuite raw camera buffers are vertically inverted.
            return {
                "agentview_image": calibration_front[::-1],
                "robot0_eye_in_hand_image": calibration_wrist[::-1],
            }

        def get_camera_meta(self, **_kwargs):
            return {
                "intrinsic_K": [[32.0, 0.0, 31.5], [0.0, 32.0, 31.5], [0.0, 0.0, 1.0]],
                "extrinsic_cam2world": np.eye(4).tolist(),
            }

    policy_front = calibration_front[:, ::-1]
    robot = SimpleNamespace(
        env=Env(),
        _last_obs={
            "main_images": policy_front,
            "wrist_images": calibration_wrist[:, ::-1],
        },
    )
    client = FakeClient()
    explorer = ParallelWorldExplorer(
        primitives=robot, client=client, tts=False,
        tts_samples=3, rollout_steps=1,
    )

    result = explorer.decide(candidates=[candidate(
        "endpoint", primitive="move_to", arguments={"xyz": [0.25, -0.25, 1.0]}
    )])

    # The clean source retained for artifacts is the calibration frame, while
    # the candidate-specific model input additionally carries marker A.
    np.testing.assert_array_equal(explorer._last_real_images[0], calibration_front)
    assert not np.array_equal(client.calls[0]["front"], policy_front)
    assert not np.array_equal(client.calls[0]["front"], calibration_front)
    instruction = client.calls[0]["instruction"]
    assert "image-left and image-right" in instruction
    assert "normalized image coordinates" in instruction
    assert "marker A" in instruction
    assert result["branches"][0]["spatial_guidance"]["normalized_uv"] is not None


def test_prompt_enhancer_output_is_sent_to_world_model():
    class Enhancer:
        def __init__(self):
            self.calls = []

        def enhance_batch(self, *, candidates, task, fronts, rollout_steps):
            self.calls.append({
                "candidates": candidates,
                "task": task,
                "fronts": {key: value.copy() for key, value in fronts.items()},
                "rollout_steps": rollout_steps,
            })
            return {
                item["id"]: {
                    "instruction": (
                        f"Enhanced route {item['id']} preserves its requested final "
                        "state through the final frame; the next task step does not begin."
                    ),
                    "source": "llm",
                    "fallback_reason": None,
                    "repair_reason": None,
                    "repair_attempted": False,
                    "structured": structured_enhancement(item["id"], "release"),
                    "word_count": 18,
                    "action_boundary_rule": "release",
                }
                for item in candidates
            }

    task = "put the lower patterned bowl on the plate"
    robot = primitives()
    robot._last_obs["task_descriptions"] = task
    enhancer = Enhancer()
    client = FakeClient()
    explorer = ParallelWorldExplorer(
        primitives=robot, client=client, tts=False,
        tts_samples=3, rollout_steps=2, prompt_enhancer=enhancer,
        task="LIBERO suite=libero_spatial_task, task=3",
    )
    routes = [candidate("route-a"), candidate("route-b")]
    result = explorer.decide(candidates=routes)

    assert len(enhancer.calls) == 1
    assert [item["id"] for item in enhancer.calls[0]["candidates"]] == [
        "route-a", "route-b",
    ]
    assert enhancer.calls[0]["task"] == task
    assert enhancer.calls[0]["rollout_steps"] == 2
    assert set(enhancer.calls[0]["fronts"]) == {"route-a", "route-b"}
    assert all(
        np.array_equal(front, robot._last_obs["main_images"])
        for front in enhancer.calls[0]["fronts"].values()
    )
    assert [call["instruction"] for call in client.calls] == [
        branch["world_model_instruction"] for branch in result["branches"]
    ]
    assert all(
        "next task step does not begin" in branch["world_model_instruction"]
        for branch in result["branches"]
    )
    assert [branch["prompt_enhancement"]["source"] for branch in result["branches"]] == [
        "llm", "llm",
    ]


def test_visual_action_schema_is_required_for_new_planner_candidates():
    item_schema = PARALLELWORLD_TOOL_SPEC["input_schema"]["properties"][
        "candidate_routes"
    ]["items"]

    assert "visual_action" in item_schema["required"]
    assert item_schema["properties"]["visual_action"]["required"] == [
        "subject",
        "motion",
    ]
    target_uv = item_schema["properties"]["visual_action"]["properties"]["target_uv"]
    assert target_uv["minItems"] == target_uv["maxItems"] == 2
    assert target_uv["items"]["minimum"] == 0.0
    assert target_uv["items"]["maximum"] == 1.0


def test_semantic_fallback_uses_grounded_visual_action_without_controls():
    route = {
        "primitive": "move_to",
        "arguments": {
            "xyz": [-0.2, 0.25, 1.1],
            "gripper": -1,
            "tol": 0.012,
            "max_steps": 100,
        },
        "visual_action": {
            "subject": "robot arm",
            "motion": "moves smoothly",
            "target": "black bowl",
            "target_qualifier": "far-right",
            "spatial_relation": "above",
            "gripper_state": "open",
            "contact": "stops before touching the bowl",
            "intended_outcome": "the gripper is safely positioned above the bowl",
        },
    }

    prompt = semantic_fallback(route)

    assert "far-right black bowl" in prompt
    assert "gripper remains open" in prompt
    assert "stops before touching" in prompt
    assert_rendered_prompt_contract(prompt)
    assert not contains_control_syntax(prompt)
    assert "-0.2" not in prompt
    assert "max_steps" not in prompt

    verbose_route = {
        **route,
        "visual_action": {
            **route["visual_action"],
            "target": "the selected rightmost black patterned bowl",
            "target_qualifier": (
                "the upper-right bowl, not the bowl beside the ramekin"
            ),
        },
    }
    verbose_prompt = semantic_fallback(verbose_route)
    assert "rightmost black patterned bowl" in verbose_prompt.lower()
    assert "upper-right the selected" not in verbose_prompt
    assert_rendered_prompt_contract(verbose_prompt)

    long_target_route = {
        **route,
        "visual_action": {
            **route["visual_action"],
            "target": (
                "the selected upper-right small glossy ceramic black patterned bowl"
            ),
            "target_qualifier": "the upper-right target bowl",
        },
    }
    long_prompt = semantic_fallback(long_target_route).lower()
    assert "upper-right small glossy ceramic black bowl" in long_prompt
    assert "the selected" not in long_prompt


def test_prompt_renderer_labels_short_fragments_without_broken_grammar():
    class Agent:
        def run_sync(self, _content):
            spec = structured_enhancement("stage", "move_to")
            spec.update({
                "initial_state": "Gripper above target bowl",
                "motion_phases": ["Moves above target area"],
                "contact_and_gripper": "Open, no contact",
                "object_response": "Remains stationary",
                "terminal_state": "Above bowl, open",
                "candidate_contrast": "Stages above target",
                "natural_instruction": (
                    "The open gripper moves above the lower patterned bowl and stops "
                    "without touching it. The bowl remains stationary throughout."
                ),
            })
            return SimpleNamespace(output=json.dumps({"candidates": [spec]}))

    route = visual_candidate(
        "stage", "move_to", gripper_state="open", arguments={"gripper": -1}
    )
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent()
    records = enhancer.enhance_batch(
        candidates=[route],
        task="approach the lower patterned bowl",
        fronts={"stage": np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )

    prompt = records["stage"]["instruction"]
    assert prompt.startswith("The open gripper moves above the lower patterned bowl")
    assert "The bowl remains stationary throughout." in prompt
    assert "initial state:" not in prompt.lower()
    assert "contact/gripper:" not in prompt.lower()
    assert "object response:" not in prompt.lower()
    assert "final state:" not in prompt.lower()
    assert_rendered_prompt_contract(prompt)


def test_prompt_renderer_keeps_maximum_valid_structure_on_llm_path():
    class Agent:
        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps({"candidates": [{
                "candidate_id": "max-bound",
                "target_identity": "lower patterned ceramic serving bowl nearby",
                "initial_state": "gripper above patterned serving bowl",
                "motion_phases": [
                    "arm moves", "above bowl", "aligns with target", "stops there",
                ],
                "contact_and_gripper": "open fingers avoid contact",
                "object_response": "bowl remains completely stationary",
                "terminal_state": "open gripper stops above bowl",
                "candidate_contrast": "highest aligned endpoint",
                "natural_instruction": (
                    "The open gripper moves above the lower patterned ceramic serving "
                    "bowl, aligns with it, and stops without contact while the bowl "
                    "remains stationary."
                ),
            }]}))

    route = visual_candidate(
        "max-bound", "move_to", gripper_state="open", arguments={"gripper": -1}
    )
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent()
    record = enhancer.enhance_batch(
        candidates=[route],
        task="approach the lower patterned bowl",
        fronts={"max-bound": np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )["max-bound"]

    assert record["source"] == "llm", record["fallback_reason"]
    assert record["fallback_reason"] is None
    assert record["word_count"] <= 90
    assert_rendered_prompt_contract(record["instruction"])


def test_prompt_enhancer_batches_candidates_with_task_horizon_and_native_front():
    from pydantic_ai import BinaryContent

    class Agent:
        def __init__(self, output):
            self.output = output
            self.calls = []

        def run_sync(self, content):
            self.calls.append(content)
            return SimpleNamespace(output=json.dumps(self.output))

    routes = [
        visual_candidate("pick-route", "pi0_pick", gripper_state="holding"),
        visual_candidate(
            "move-route", "move_to", gripper_state="holding",
            arguments={"gripper": 1},
        ),
    ]
    # Deliberately reverse the model response: callers must still receive records
    # in planner-candidate order, keyed by candidate_id rather than response order.
    response = {
        "candidates": [
            structured_enhancement("move-route", "move_to"),
            structured_enhancement("pick-route", "pi0_pick"),
        ]
    }
    pick_front = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    move_front = np.full((9, 13, 3), 37, dtype=np.uint8)
    fronts = {"pick-route": pick_front, "move-route": move_front}
    task = "pick up the lower patterned bowl beside the cookie package"
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent(response)

    records = enhancer.enhance_batch(
        candidates=routes, task=task, fronts=fronts, rollout_steps=49,
    )

    assert len(enhancer._agent.calls) == 1
    content = enhancer._agent.calls[0]
    assert isinstance(content, list)
    json_items = []
    for item in content:
        if isinstance(item, str):
            try:
                json_items.append(json.loads(item))
            except json.JSONDecodeError:
                pass
    assert len(json_items) == 1
    payload = json_items[0]
    assert payload["task"] == task
    assert "rollout_steps" not in payload
    assert payload["prediction_target"] == "single_terminal_state_image"
    assert payload["candidate_ids"] == ["pick-route", "move-route"]
    assert [item["id"] for item in payload["candidates"]] == [
        "pick-route", "move-route",
    ]

    text_items = [item for item in content if isinstance(item, str)]
    assert "Candidate ID: pick-route; current front view." in text_items
    assert "Candidate ID: move-route; current front view." in text_items
    assert all("wrist" not in item.lower() for item in text_items)
    binary_items = [item for item in content if isinstance(item, BinaryContent)]
    assert len(binary_items) == len(routes)
    assert all(item.media_type == "image/png" for item in binary_items)
    assert all(item.vendor_metadata == {"detail": "high"} for item in binary_items)
    for item, candidate_id in zip(binary_items, ("pick-route", "move-route")):
        assert candidate_id in str(item.identifier)
        decoded = np.asarray(Image.open(io.BytesIO(item.data)).convert("RGB"))
        assert decoded.shape == fronts[candidate_id].shape
        assert np.array_equal(decoded, fronts[candidate_id])

    assert list(records) == ["pick-route", "move-route"]
    assert [records[candidate_id]["structured"]["candidate_id"] for candidate_id in records] == [
        "pick-route", "move-route",
    ]
    required_fields = {
        "candidate_id", "target_identity", "initial_state", "motion_phases",
        "contact_and_gripper", "object_response", "terminal_state",
        "candidate_contrast", "natural_instruction",
    }
    for record in records.values():
        assert (
            record["prompt_enhancement_version"]
            == PROMPT_ENHANCEMENT_VERSION
            == "v8"
        )
        assert record["source"] == "llm"
        assert record["fallback_reason"] is None
        assert record["repair_attempted"] is False
        assert required_fields <= set(record["structured"])
        assert record["word_count"] == len(re.findall(
            r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", record["instruction"]
        ))
        assert_rendered_prompt_contract(record["instruction"])


def test_prompt_enhancer_accepts_direct_string_agent_output():
    class Agent:
        def __init__(self):
            self.calls = 0

        def run_sync(self, _content):
            self.calls += 1
            return json.dumps({
                "candidates": [
                    structured_enhancement("direct-string", "release")
                ]
            })

    route = visual_candidate("direct-string", "release", gripper_state="open")
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent()

    record = enhancer.enhance_batch(
        candidates=[route],
        task="release the lower patterned bowl onto the support",
        fronts={"direct-string": np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )["direct-string"]

    assert enhancer._agent.calls == 1
    assert record["source"] == "llm"
    assert record["fallback_reason"] is None
    assert record["prompt_enhancement_version"] == "v8"
    assert_rendered_prompt_contract(record["instruction"])


def test_prompt_enhancer_accepts_direct_string_repair_output():
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = 0

        def run_sync(self, _content):
            self.calls += 1
            output = self.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return json.dumps(output)

    valid = structured_enhancement("direct-repair", "release")
    agent = Agent([
        {"candidates": []},
        {"candidates": [valid]},
    ])
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = agent
    route = visual_candidate("direct-repair", "release", gripper_state="open")

    record = enhancer.enhance_batch(
        candidates=[route],
        task="release the lower patterned bowl onto the support",
        fronts={"direct-repair": np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )["direct-repair"]

    assert agent.calls == 2
    assert record["source"] == "llm_repair"
    assert record["repair_attempted"] is True
    assert "candidate IDs must match" in record["repair_reason"]
    assert record["fallback_reason"] is None
    assert record["prompt_enhancement_version"] == "v8"
    assert_rendered_prompt_contract(record["instruction"])


def test_prompt_enhancer_repairs_invalid_group_exactly_once():
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []

        def run_sync(self, content):
            self.calls.append(content)
            return SimpleNamespace(output=json.dumps(self.outputs.pop(0)))

    route_a = visual_candidate("pick-route", "pi0_pick", gripper_state="holding")
    route_b = visual_candidate("release-route", "release", gripper_state="open")
    invalid_b = structured_enhancement("release-route", "release")
    invalid_b["terminal_state"] = ""
    repaired_b = structured_enhancement("release-route", "release")
    agent = Agent([
        {"candidates": [structured_enhancement("pick-route", "pi0_pick"), invalid_b]},
        {"candidates": [
            structured_enhancement("pick-route", "pi0_pick"), repaired_b,
        ]},
    ])
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = agent

    records = enhancer.enhance_batch(
        candidates=[route_a, route_b], task="move the patterned bowl",
        fronts={
            "pick-route": np.zeros((7, 11, 3), dtype=np.uint8),
            "release-route": np.ones((9, 13, 3), dtype=np.uint8),
        },
        rollout_steps=49,
    )

    assert len(agent.calls) == 2
    assert [record["source"] for record in records.values()] == [
        "llm_repair", "llm_repair",
    ]
    assert all(record["repair_attempted"] is True for record in records.values())
    assert all(record["repair_reason"] for record in records.values())
    assert all(record["fallback_reason"] is None for record in records.values())
    assert records["release-route"]["structured"]["terminal_state"] == repaired_b["terminal_state"]
    assert_rendered_prompt_contract(records["release-route"]["instruction"])


def test_prompt_enhancer_accepts_target_references_but_repairs_mirrored_side():
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps(self.outputs.pop(0)))

    route_a = visual_candidate(
        "literal-target", "move_to", gripper_state="open", arguments={"gripper": -1}
    )
    route_a["visual_action"]["target"] = "the upper-right black patterned bowl"
    route_b = visual_candidate(
        "noun-reference", "move_to", gripper_state="open", arguments={"gripper": -1}
    )
    route_b["visual_action"]["target"] = "the far-right dark object"

    def stage_spec(candidate_id, target_identity, initial_state, contrast):
        return {
            "candidate_id": candidate_id,
            "target_identity": target_identity,
            "initial_state": initial_state,
            "motion_phases": ["Moves above target"],
            "contact_and_gripper": "Open, no contact",
            "object_response": "Bowl remains stationary",
            "terminal_state": "Gripper stops above bowl",
            "candidate_contrast": contrast,
            "natural_instruction": (
                f"The open gripper moves toward the {target_identity} and stops above "
                "it without contact. The bowl remains stationary at the endpoint."
            ),
        }

    wrong_a = stage_spec(
        "literal-target", "upper-left black patterned bowl",
        "Gripper right of target, open", "direct staged endpoint",
    )
    correct_a = stage_spec(
        "literal-target", "upper-right black patterned bowl",
        "Gripper right of target, open", "direct staged endpoint",
    )
    correct_b = stage_spec(
        "noun-reference", "far-right dark patterned bowl",
        "Gripper above-right of bowl", "higher staged endpoint",
    )
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent([
        {"candidates": [wrong_a, correct_b]},
        {"candidates": [correct_a, correct_b]},
    ])

    records = enhancer.enhance_batch(
        candidates=[route_a, route_b],
        task="approach the upper-right patterned bowl",
        fronts={
            "literal-target": np.zeros((7, 11, 3), dtype=np.uint8),
            "noun-reference": np.ones((7, 11, 3), dtype=np.uint8),
        },
        rollout_steps=49,
    )

    assert [record["source"] for record in records.values()] == [
        "llm_repair", "llm_repair",
    ]
    assert all(record["fallback_reason"] is None for record in records.values())
    assert "right-side qualifier" in records["literal-target"]["repair_reason"]
    assert records["literal-target"]["structured"]["target_identity"].startswith(
        "upper-right"
    )


def test_prompt_enhancer_falls_back_with_reason_after_one_failed_repair():
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []

        def run_sync(self, content):
            self.calls.append(content)
            return SimpleNamespace(
                output=json.dumps(self.outputs.pop(0))
            )

    route = visual_candidate("pick-route", "pi0_pick", gripper_state="holding")
    invalid = structured_enhancement("pick-route", "pi0_pick")
    invalid["object_response"] = ""
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent([
        {"candidates": [invalid]},
        {"candidates": [invalid]},
    ])

    records = enhancer.enhance_batch(
        candidates=[route], task="pick up the patterned bowl",
        fronts={"pick-route": np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )

    assert len(enhancer._agent.calls) == 2
    record = records["pick-route"]
    assert (
        record["prompt_enhancement_version"]
        == PROMPT_ENHANCEMENT_VERSION
            == "v8"
    )
    assert record["source"] == "semantic_fallback"
    assert record["repair_attempted"] is True
    assert record["repair_reason"]
    assert record["fallback_reason"]
    assert "object_response" in (
        record["repair_reason"] + " " + record["fallback_reason"]
    )
    assert_rendered_prompt_contract(record["instruction"])


def test_prompt_renderer_uses_action_aware_terminal_states_without_extra_steps():
    class Agent:
        def __init__(self, output):
            self.output = output

        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps(self.output))

    routes = [
        visual_candidate("pick", "pi0_pick", gripper_state="holding"),
        visual_candidate(
            "move", "move_to", gripper_state="holding", arguments={"gripper": 1},
        ),
        visual_candidate("release", "release", gripper_state="open"),
        visual_candidate(
            "close", "set_gripper", gripper_state="closed",
            arguments={"gripper": 1},
        ),
    ]
    response = {"candidates": [
        structured_enhancement("pick", "pi0_pick"),
        structured_enhancement("move", "move_to"),
        structured_enhancement("release", "release"),
        structured_enhancement("close", "set_gripper_close"),
    ]}
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent(response)

    records = enhancer.enhance_batch(
        candidates=routes, task="manipulate the lower patterned bowl",
        fronts={
            route["id"]: np.full((7, 11, 3), index, dtype=np.uint8)
            for index, route in enumerate(routes)
        },
        rollout_steps=49,
    )
    prompts = {
        candidate_id: record["instruction"]
        for candidate_id, record in records.items()
    }

    for prompt in prompts.values():
        assert_rendered_prompt_contract(prompt)
    concrete_follow_up = re.compile(
        r"\b(?:release(?:s|d|ing)?|plac(?:e|es|ed|ing)|set\s+down)\b",
        re.IGNORECASE,
    )
    assert concrete_follow_up.search(prompts["pick"]) is None
    assert concrete_follow_up.search(prompts["move"]) is None
    assert re.search(r"\breleas(?:e|es|ed|ing)\b", prompts["release"], re.I)
    assert re.search(r"\b(?:rest|rests|resting|settle|settles|settled)\b", prompts["release"], re.I)
    assert "open gripper" in prompts["release"].lower()
    assert "closed" in prompts["close"].lower()
    assert "arm" in prompts["close"].lower()


@pytest.mark.parametrize(
    "sample",
    [
        pytest.param({
            "primitive": "pi0_pick",
            "arguments": {"prompt": "pick up the lower patterned bowl"},
            "gripper_state": "holding",
            "target": "lower patterned bowl",
            "motion": "descends, closes around the rim, and lifts the bowl clear",
            "phases": ["descend to bowl", "close fingers securely", "lift bowl clear"],
            "contact": "fingers close around rim",
            "response": "bowl follows gripper",
            "terminal": "bowl remains securely held",
            "instruction": (
                "The robot gripper descends to the lower patterned bowl, closes securely "
                "around its rim, and lifts it clear with the bowl held steadily at the "
                "reached pose."
            ),
            "boundary": "pick_hold",
            "required": (r"robot gripper", r"closes", r"lifts", r"securely held"),
            "forbidden": (r"\breleas", r"\bplac"),
        }, id="pi0_pick"),
        pytest.param({
            "primitive": "pi0_doubled",
            "arguments": {"prompt": "press the front red stove button"},
            "gripper_state": "unchanged",
            "target": "front red stove button",
            "motion": "presses the button and stops at the immediate result",
            "phases": ["robot presses button once"],
            "contact": "finger contacts button",
            "response": "button depresses naturally",
            "terminal": "robot stops at result",
            "instruction": (
                "The robot gripper presses the front red stove button once. The button "
                "depresses naturally, and the robot stops as soon as the immediate visible "
                "contact result is reached."
            ),
            "boundary": "contact_hold",
            "required": (r"robot gripper", r"presses", r"immediate visible", r"stops"),
            "forbidden": (r"\bgrasp", r"\blift", r"\bplac"),
        }, id="pi0_doubled"),
        pytest.param({
            "primitive": "move_to",
            "arguments": {"gripper": -1},
            "gripper_state": "open",
            "target": "far-right patterned black bowl",
            "motion": "moves smoothly toward and above the bowl",
            "phases": ["gripper approaches bowl", "stops above target"],
            "contact": "open gripper avoids contact",
            "response": "bowl remains stationary",
            "terminal": "open gripper stops above bowl",
            "instruction": (
                "The open robot gripper moves smoothly toward the far-right patterned "
                "black bowl and stops above it without making contact. The bowl stays "
                "in place throughout."
            ),
            "boundary": "move_open",
            "required": (r"open robot gripper moves", r"without making contact", r"bowl stays in place"),
            "forbidden": (r"^Move the .*bowl", r"\bheld bowl", r"\bcarries the bowl"),
        }, id="move_to_open"),
        pytest.param({
            "primitive": "move_to",
            "arguments": {"gripper": 1},
            "gripper_state": "holding",
            "target": "white plate with red rings",
            "motion": "carries the held bowl smoothly above the plate",
            "phases": ["carry held bowl smoothly", "stop above plate"],
            "contact": "closed grasp remains stable",
            "response": "held bowl follows gripper",
            "terminal": "bowl and gripper stop together",
            "instruction": (
                "The robot gripper carries the securely held patterned bowl smoothly "
                "toward the white plate with red rings and stops above it without "
                "changing the closed grasp."
            ),
            "boundary": "move_holding",
            "required": (r"robot gripper carries", r"securely held", r"closed grasp", r"held object follows the gripper rigidly", r"stop"),
            "forbidden": (r"\breleas", r"\bplac"),
        }, id="move_to_holding"),
        pytest.param({
            "primitive": "move_pose",
            "arguments": {"gripper": -1},
            "gripper_state": "open",
            "target": "lower shelf opening",
            "motion": "moves toward the shelf while adjusting wrist orientation",
            "phases": ["gripper translates toward shelf", "wrist reorients simultaneously"],
            "contact": "open gripper avoids contact",
            "response": "shelf remains stationary",
            "terminal": "open gripper stops aligned",
            "instruction": (
                "The open robot gripper moves toward the lower shelf opening while "
                "changing its wrist orientation simultaneously, then stops aligned "
                "while maintaining safe clearance. The shelf remains stationary throughout."
            ),
            "boundary": "move_open",
            "required": (r"open robot gripper moves", r"wrist orientation", r"maintaining safe clearance", r"shelf remains stationary"),
            "forbidden": (r"^Move the .*shelf", r"\bheld shelf"),
        }, id="move_pose_open"),
        pytest.param({
            "primitive": "move_pose",
            "arguments": {"gripper": 1},
            "gripper_state": "holding",
            "target": "white plate with red rings",
            "motion": "carries the held bowl while changing wrist orientation",
            "phases": ["carry held bowl forward", "wrist reorients simultaneously"],
            "contact": "closed grasp remains stable",
            "response": "held bowl follows gripper",
            "terminal": "bowl and gripper stop together",
            "instruction": (
                "The robot gripper carries the securely held patterned bowl toward the "
                "white plate while changing position and wrist orientation together, "
                "then stops with the closed grasp unchanged."
            ),
            "boundary": "move_holding",
            "required": (r"robot gripper carries", r"wrist orientation", r"closed grasp", r"held object follows the gripper's position and orientation rigidly", r"stops"),
            "forbidden": (r"\breleas", r"\bplac"),
        }, id="move_pose_holding"),
        pytest.param({
            "primitive": "release",
            "arguments": {},
            "gripper_state": "open",
            "target": "white plate with red rings",
            "motion": "opens in place and releases the held bowl onto the plate",
            "phases": ["gripper opens at current pose"],
            "contact": "fingers open over support",
            "response": "released bowl settles naturally",
            "terminal": "bowl rests gripper stays open",
            "instruction": (
                "The robot gripper opens at its current pose and releases the held bowl "
                "onto the white plate with red rings. The bowl settles naturally while "
                "the open gripper remains in place."
            ),
            "boundary": "release_settle",
            "required": (r"robot gripper opens", r"releases", r"settles naturally", r"remains in place"),
            "forbidden": (r"\bwithdraw", r"\bregrasp", r"moves away"),
        }, id="release"),
        pytest.param({
            "primitive": "set_gripper",
            "arguments": {"gripper": 1},
            "gripper_state": "closed",
            "target": "lower patterned bowl",
            "motion": "closes its fingers around the bowl without moving the arm",
            "phases": ["close gripper with arm fixed"],
            "contact": "fingers maintain current grasp",
            "response": "held bowl remains steady",
            "terminal": "closed gripper arm stays fixed",
            "instruction": (
                "The stationary robot gripper closes securely around the lower patterned "
                "bowl. The arm remains fixed, the bowl stays steady, and the resulting "
                "grasp remains closed."
            ),
            "boundary": "close_in_place",
            "required": (r"robot gripper closes", r"arm remains fixed", r"bowl stays steady", r"grasp remains closed"),
            "forbidden": (r"moves toward", r"\breleas"),
        }, id="set_gripper_close"),
        pytest.param({
            "primitive": "set_gripper",
            "arguments": {"gripper": -1},
            "gripper_state": "open",
            "target": "lower patterned bowl",
            "motion": "opens its fingers without moving the arm",
            "phases": ["open gripper with arm fixed"],
            "contact": "fingers end current contact",
            "response": "supported bowl settles naturally",
            "terminal": "open gripper arm stays fixed",
            "instruction": (
                "The robot gripper opens around the lower patterned bowl without moving "
                "the arm. The supported bowl settles naturally while the gripper remains "
                "at the same arm pose."
            ),
            "boundary": "open_in_place",
            "required": (r"robot gripper opens", r"without moving the arm", r"settles naturally", r"same arm pose"),
            "forbidden": (r"moves toward", r"securely held"),
        }, id="set_gripper_open"),
        pytest.param({
            "primitive": "rotate_wrist",
            "arguments": {"gripper": 1},
            "gripper_state": "holding",
            "target": "held lower patterned bowl",
            "motion": "rotates smoothly in place with the held bowl",
            "phases": ["wrist rotates smoothly in place"],
            "contact": "gripper preserves closed grasp",
            "response": "held bowl rotates rigidly",
            "terminal": "wrist stops grasp unchanged",
            "instruction": (
                "The robot wrist rotates smoothly in place while the gripper keeps the "
                "held lower patterned bowl securely grasped. The arm position and closed "
                "grasp remain unchanged."
            ),
            "boundary": "rotate_holding",
            "required": (r"robot wrist rotates", r"securely grasped", r"held object rotates rigidly", r"position", r"closed grasp"),
            "forbidden": (r"moves toward", r"\breleas"),
        }, id="rotate_wrist_holding"),
        pytest.param({
            "primitive": "rotate_wrist",
            "arguments": {"gripper": -1},
            "gripper_state": "open",
            "target": "lower patterned bowl",
            "motion": "rotates smoothly in place beside the bowl",
            "phases": ["wrist rotates smoothly in place"],
            "contact": "open gripper avoids contact",
            "response": "bowl remains stationary",
            "terminal": "open wrist stops in place",
            "instruction": (
                "The robot wrist rotates smoothly in place near the lower patterned bowl "
                "while the gripper remains open, without touching the bowl or translating "
                "the arm. The bowl remains stationary."
            ),
            "boundary": "rotate_open",
            "required": (r"robot wrist rotates", r"gripper remains open", r"without touching", r"translating the arm", r"bowl remains stationary", r"target object remains stationary"),
            "forbidden": (r"securely held", r"closed grasp", r"moves toward"),
        }, id="rotate_wrist_open"),
        pytest.param({
            "primitive": "rotate_pitch",
            "arguments": {"gripper": 1},
            "gripper_state": "holding",
            "target": "held lower patterned bowl",
            "motion": "tilts smoothly in place with the held bowl",
            "phases": ["wrist tilts smoothly in place"],
            "contact": "gripper preserves closed grasp",
            "response": "held bowl tilts rigidly",
            "terminal": "wrist stops grasp unchanged",
            "instruction": (
                "The robot wrist tilts smoothly in place while the gripper keeps the held "
                "lower patterned bowl securely grasped. The arm position and closed "
                "grasp remain unchanged."
            ),
            "boundary": "rotate_holding",
            "required": (r"robot wrist tilts", r"securely grasped", r"held object tilts rigidly", r"position", r"closed grasp"),
            "forbidden": (r"moves toward", r"\breleas"),
        }, id="rotate_pitch_holding"),
        pytest.param({
            "primitive": "rotate_pitch",
            "arguments": {"gripper": -1},
            "gripper_state": "open",
            "target": "lower patterned bowl",
            "motion": "tilts smoothly in place beside the bowl",
            "phases": ["wrist tilts smoothly in place"],
            "contact": "open gripper avoids contact",
            "response": "bowl remains stationary",
            "terminal": "open wrist stops in place",
            "instruction": (
                "The robot wrist tilts smoothly in place near the lower patterned bowl "
                "while the gripper remains open, without touching the bowl or translating "
                "the arm. The bowl remains stationary."
            ),
            "boundary": "rotate_open",
            "required": (r"robot wrist tilts", r"gripper remains open", r"without touching", r"translating the arm", r"bowl remains stationary", r"target object remains stationary"),
            "forbidden": (r"securely held", r"closed grasp", r"moves toward"),
        }, id="rotate_pitch_open"),
    ],
)
def test_prompt_semantics_for_each_executable_action(sample):
    candidate_id = f"sample-{sample['primitive']}"
    route = {
        "id": candidate_id,
        "primitive": sample["primitive"],
        "arguments": sample["arguments"],
        "rationale": "exercise one bounded visible action",
        "visual_action": {
            "subject": "robot gripper",
            "motion": sample["motion"],
            "target": sample["target"],
            "target_qualifier": sample["target"],
            "spatial_relation": "at the immediate target",
            "gripper_state": sample["gripper_state"],
            "contact": sample["contact"],
            "intended_outcome": sample["terminal"],
        },
    }
    spec = {
        "candidate_id": candidate_id,
        "target_identity": sample["target"],
        "initial_state": "gripper starts near target",
        "motion_phases": sample["phases"],
        "contact_and_gripper": sample["contact"],
        "object_response": sample["response"],
        "terminal_state": sample["terminal"],
        "candidate_contrast": "bounded visible endpoint",
        "natural_instruction": sample["instruction"],
    }

    class Agent:
        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps({"candidates": [spec]}))

    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent()
    record = enhancer.enhance_batch(
        candidates=[route],
        task="perform exactly the candidate action",
        fronts={candidate_id: np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )[candidate_id]

    assert record["source"] == "llm", record["fallback_reason"]
    assert record["fallback_reason"] is None
    assert record["action_boundary_rule"] == sample["boundary"]
    assert_rendered_prompt_contract(record["instruction"])
    for pattern in sample["required"]:
        assert re.search(pattern, record["instruction"], re.IGNORECASE), pattern
    for pattern in sample["forbidden"]:
        assert not re.search(pattern, record["instruction"], re.IGNORECASE), pattern

    fallback = semantic_fallback(route)
    assert_rendered_prompt_contract(fallback)
    for pattern in sample["forbidden"]:
        assert not re.search(pattern, fallback, re.IGNORECASE), pattern
    primitive = sample["primitive"]
    mode = sample["gripper_state"]
    if primitive == "pi0_pick":
        assert re.search(r"robot gripper.*closes.*lifts", fallback, re.IGNORECASE)
        assert re.search(r"securely held", fallback, re.IGNORECASE)
    elif primitive == "pi0_doubled":
        assert re.search(r"robot gripper.*presses", fallback, re.IGNORECASE)
        assert re.search(r"immediate visible contact result", fallback, re.IGNORECASE)
    elif primitive in {"move_to", "move_pose"}:
        assert re.search(r"robot gripper (?:moves|carries)", fallback, re.IGNORECASE)
        if mode == "open":
            assert re.search(r"gripper remains open", fallback, re.IGNORECASE)
            assert re.search(r"without touching|avoids contact", fallback, re.IGNORECASE)
            assert re.search(r"target remains stationary", fallback, re.IGNORECASE)
        else:
            assert re.search(r"closed grasp", fallback, re.IGNORECASE)
            if primitive == "move_pose":
                assert re.search(
                    r"held object follows the gripper's position and orientation rigidly",
                    fallback,
                    re.IGNORECASE,
                )
            else:
                assert re.search(
                    r"held object follows the gripper rigidly", fallback, re.IGNORECASE
                )
        if primitive == "move_pose":
            assert re.search(r"position and (?:wrist )?orientation", fallback, re.IGNORECASE)
    elif primitive == "release":
        assert re.search(r"robot gripper opens.*releases", fallback, re.IGNORECASE)
        assert re.search(r"settles naturally", fallback, re.IGNORECASE)
    elif primitive == "set_gripper":
        assert re.search(r"robot gripper (?:opens|closes)", fallback, re.IGNORECASE)
        assert re.search(r"arm stays fixed|without moving the arm", fallback, re.IGNORECASE)
        if mode == "holding":
            assert re.search(r"target remains steady", fallback, re.IGNORECASE)
    elif primitive in {"rotate_wrist", "rotate_pitch"}:
        motion = "tilts" if primitive == "rotate_pitch" else "rotates"
        assert re.search(rf"robot wrist {motion}.*in place", fallback, re.IGNORECASE)
        if mode == "holding":
            assert re.search(
                rf"held object {motion} rigidly with the wrist", fallback, re.IGNORECASE
            )
        else:
            assert re.search(r"gripper remains open", fallback, re.IGNORECASE)
            assert re.search(r"target remains stationary", fallback, re.IGNORECASE)


@pytest.mark.parametrize(
    ("bad_instruction", "expected_error"),
    [
        pytest.param(
            "Move the far-right patterned black bowl outside the plate-to-ramekin line, "
            "keeping the gripper open and clear of contact above it.",
            "stationary target the direct object",
            id="named-target-as-direct-object",
        ),
        pytest.param(
            "The gripper is open; the bowl moves toward the waypoint without contact, "
            "then the far-right patterned black bowl remains stationary at the endpoint.",
            "stationary target the moving subject",
            id="named-target-as-subject",
        ),
        pytest.param(
            "The open robot gripper moves it toward the line without contact, although "
            "the far-right patterned black bowl remains stationary at the endpoint.",
            "stationary target the direct object",
            id="target-pronoun-as-direct-object",
        ),
    ],
)
def test_prompt_enhancer_repairs_move_open_that_moves_the_target_object(
    bad_instruction, expected_error
):
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []

        def run_sync(self, content):
            self.calls.append(content)
            return SimpleNamespace(output=json.dumps(self.outputs.pop(0)))

    route = {
        "id": "approach-coarse-move",
        "primitive": "move_to",
        "arguments": {"gripper": -1},
        "rationale": "stage the arm above the selected bowl",
        "visual_action": {
            "subject": "the robot gripper",
            "motion": "moves smoothly to a high position above the target area",
            "target": "far-right patterned black bowl",
            "target_qualifier": "outside the plate-to-ramekin line",
            "spatial_relation": "toward and above",
            "gripper_state": "open",
            "contact": "does not touch any object",
            "intended_outcome": "gripper stops above the selected bowl",
        },
    }
    bad = {
        "candidate_id": route["id"],
        "target_identity": "far-right patterned black bowl",
        "initial_state": "gripper starts above target",
        "motion_phases": ["translate toward target", "stage high above"],
        "contact_and_gripper": "open with no contact",
        "object_response": "bowl remains stationary",
        "terminal_state": "open gripper stops above bowl",
        "candidate_contrast": "coarse staged endpoint",
        "natural_instruction": bad_instruction,
    }
    repaired = {
        **bad,
        "natural_instruction": (
            "The open robot gripper moves toward the far-right patterned black bowl and "
            "stops above it without touching it. The bowl remains stationary throughout."
        ),
    }
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent([
        {"candidates": [bad]},
        {"candidates": [repaired]},
    ])

    record = enhancer.enhance_batch(
        candidates=[route],
        task="approach the far-right patterned black bowl",
        fronts={route["id"]: np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )[route["id"]]

    assert len(enhancer._agent.calls) == 2
    assert record["source"] == "llm_repair"
    assert record["repair_attempted"] is True
    assert expected_error in record["repair_reason"]
    assert record["fallback_reason"] is None
    assert record["instruction"].startswith("The open robot gripper moves toward")
    assert "bowl remains stationary" in record["instruction"].lower()
    assert not re.search(r"^Move the .*bowl", record["instruction"], re.IGNORECASE)
    assert_rendered_prompt_contract(record["instruction"])


@pytest.mark.parametrize(
    ("primitive", "motion", "response", "bad_response", "instruction", "bad_instruction"),
    [
        pytest.param(
            "move_to",
            "carries the held bowl toward the plate and stops",
            "held bowl follows gripper",
            "held bowl moves independently",
            "The robot gripper carries the securely held bowl toward the plate and stops "
            "at the waypoint with the closed grasp unchanged.",
            "The robot gripper carries the securely held bowl toward the plate, but the "
            "held bowl swings loosely before stopping with the closed grasp.",
            id="move-holding",
        ),
        pytest.param(
            "rotate_wrist",
            "rotates smoothly in place with the held bowl",
            "held bowl rotates rigidly",
            "held bowl swings freely",
            "The robot wrist rotates smoothly in place while the gripper keeps the bowl "
            "securely held. The arm position and closed grasp remain unchanged.",
            "The robot wrist rotates in place while the securely held bowl swings freely, "
            "and the closed grasp remains unchanged.",
            id="rotation-holding",
        ),
    ],
)
def test_prompt_enhancer_repairs_nonrigid_motion_of_a_held_object(
    primitive, motion, response, bad_response, instruction, bad_instruction
):
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps(self.outputs.pop(0)))

    route = {
        "id": "held-object-motion",
        "primitive": primitive,
        "arguments": {"gripper": 1},
        "visual_action": {
            "subject": "robot gripper",
            "motion": motion,
            "target": "lower patterned bowl",
            "target_qualifier": "lower patterned bowl",
            "gripper_state": "holding",
            "contact": "closed grasp remains stable",
            "intended_outcome": "held bowl stops with gripper",
        },
    }
    repaired = {
        "candidate_id": route["id"],
        "target_identity": "lower patterned bowl",
        "initial_state": "gripper starts near bowl",
        "motion_phases": [motion],
        "contact_and_gripper": "closed grasp remains stable",
        "object_response": response,
        "terminal_state": "bowl and gripper stop together",
        "candidate_contrast": "bounded held endpoint",
        "natural_instruction": instruction,
    }
    bad = {
        **repaired,
        "object_response": bad_response,
        "natural_instruction": bad_instruction,
    }
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent([
        {"candidates": [bad]},
        {"candidates": [repaired]},
    ])

    record = enhancer.enhance_batch(
        candidates=[route],
        task="keep the held bowl rigidly attached to the gripper",
        fronts={route["id"]: np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )[route["id"]]

    assert record["source"] == "llm_repair", record["fallback_reason"]
    assert "independently or loosely" in record["repair_reason"] or (
        "slip, swing, or move independently" in record["repair_reason"]
    )
    assert "contradicts the rigidly held object motion" in record["repair_reason"]
    assert not re.search(
        r"independent(?:ly)?|loose(?:ly)?|swing(?:s|ing)?|swung",
        record["instruction"],
        re.IGNORECASE,
    )
    assert_rendered_prompt_contract(record["instruction"])


def test_rotation_repair_does_not_hide_an_extra_move_after_a_negated_constraint():
    class Agent:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps(self.outputs.pop(0)))

    route = {
        "id": "bounded-wrist-rotation",
        "primitive": "rotate_wrist",
        "arguments": {"gripper": 1},
        "visual_action": {
            "subject": "robot wrist",
            "motion": "rotates in place with the held bowl",
            "target": "lower patterned bowl",
            "target_qualifier": "lower patterned bowl",
            "gripper_state": "holding",
            "contact": "closed grasp remains stable",
            "intended_outcome": "wrist stops at final orientation",
        },
    }
    repaired = {
        "candidate_id": route["id"],
        "target_identity": "lower patterned bowl",
        "initial_state": "gripper starts near bowl",
        "motion_phases": ["wrist rotates smoothly in place"],
        "contact_and_gripper": "closed grasp remains stable",
        "object_response": "held bowl rotates rigidly",
        "terminal_state": "wrist stops grasp unchanged",
        "candidate_contrast": "bounded wrist endpoint",
        "natural_instruction": (
            "The robot wrist rotates smoothly in place while the gripper keeps the lower "
            "patterned bowl securely held. The arm position and grasp remain unchanged."
        ),
    }
    bad = {
        **repaired,
        "natural_instruction": (
            "The robot wrist rotates in place without releasing the securely held bowl and "
            "then moves toward the plate while keeping the closed grasp."
        ),
    }
    enhancer = object.__new__(PromptEnhancer)
    enhancer._agent = Agent([
        {"candidates": [bad]},
        {"candidates": [repaired]},
    ])

    record = enhancer.enhance_batch(
        candidates=[route],
        task="rotate the held bowl in place",
        fronts={route["id"]: np.zeros((7, 11, 3), dtype=np.uint8)},
        rollout_steps=49,
    )[route["id"]]

    assert record["source"] == "llm_repair", record["fallback_reason"]
    assert "translates the arm during an in-place rotation" in record["repair_reason"]
    assert "moves toward the plate" not in record["instruction"].lower()
    assert_rendered_prompt_contract(record["instruction"])



def test_verifier_marks_only_selected_candidate_as_accepted():
    class Verifier:
        def decide(self, **_kwargs):
            return {
                "keep": ["b"], "dominant_id": "b",
                "continue_search": False, "reason": "b is better",
            }

    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=FakeClient(), tts=False,
        tts_samples=3, rollout_steps=1, verifier=Verifier(),
    )
    result = explorer.decide(candidates=[candidate("a"), candidate("b")])

    assert [branch["id"] for branch in result["branches"]] == ["a", "b"]
    assert [branch["accepted"] for branch in result["branches"]] == [False, True]


def test_explorer_samples_front_every_ten_frames_and_appends_final_frame():
    class SequencedClient(FakeClient):
        def rollout(self, **kwargs):
            self.calls.append(kwargs)
            horizon = kwargs["horizon"]
            front = np.empty((1, horizon, 6, 10, 3), dtype=np.uint8)
            for frame_index in range(horizon):
                front[:, frame_index] = frame_index
            return {"front": front}

    class Verifier:
        def __init__(self):
            self.calls = []

        def decide(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "keep": ["route-a"], "dominant_id": "route-a",
                "continue_search": False, "reason": "selected",
            }

    verifier = Verifier()
    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=SequencedClient(), tts=False,
        tts_samples=3, rollout_steps=49, verifier=verifier,
    )

    explorer.decide(candidates=[candidate("route-a"), candidate("route-b")])

    call = verifier.calls[0]
    assert "videos" not in call
    assert "images" not in call
    wrist_frames = [frame for frame in call["frames"] if frame["view"] == "wrist_input"]
    assert [frame["candidate_id"] for frame in wrist_frames] == ["route-a", "route-b"]
    assert all(frame["view"] != "wrist_generated" for frame in call["frames"])
    assert all("wrist" not in video for video in explorer._last_rollout_videos)
    frames = [frame for frame in call["frames"] if frame["view"] == "front"]
    assert [
        (frame["candidate_id"], frame["frame_index"], frame["view"])
        for frame in frames
    ] == [
        (candidate_id, frame_index, "front")
        for candidate_id in ("route-a", "route-b")
        for frame_index in (0, 10, 20, 30, 40, 48)
    ]
    for frame in frames:
        assert frame["image"].shape == (6, 10, 3)
        assert np.all(frame["image"] == frame["frame_index"])


def test_explorer_does_not_duplicate_terminal_verifier_frame():
    class Verifier:
        def __init__(self):
            self.frame_indices = []

        def decide(self, **kwargs):
            self.frame_indices.append([
                frame["frame_index"] for frame in kwargs["frames"] if frame["view"] == "front"
            ])
            return {
                "keep": ["route-a"], "dominant_id": "route-a",
                "continue_search": False, "reason": "selected",
            }

    verifier = Verifier()
    for horizon, expected in ((1, [0]), (11, [0, 10]), (12, [0, 10, 11])):
        explorer = ParallelWorldExplorer(
            primitives=primitives(), client=FakeClient(), tts=False,
            tts_samples=3, rollout_steps=horizon, verifier=verifier,
        )
        explorer.decide(candidates=[candidate("route-a")])
        assert verifier.frame_indices[-1] == expected


def test_verifier_sends_labelled_native_resolution_png_frames():
    from pydantic_ai import BinaryContent

    class Agent:
        def __init__(self):
            self.content = None

        def run_sync(self, content):
            self.content = content
            return SimpleNamespace(output=(
                '{"keep":["route-a"],"dominant_id":"route-a",'
                '"continue_search":false,"reason":"selected"}'
            ))

    source_front = np.full((7, 11, 3), 17, dtype=np.uint8)
    agent = Agent()
    verifier = object.__new__(VerifierAgent)
    verifier._agent = agent

    verifier.decide(
        depth=1,
        candidates=[candidate("route-a")],
        frames=[
            {"candidate_id": "route-a", "frame_index": 10,
             "view": "front", "image": source_front},
        ],
    )

    assert isinstance(agent.content, list)
    text_items = [item for item in agent.content if isinstance(item, str)]
    assert "not videos" in text_items[0]
    assert "wrist_input" in text_items[0]
    assert "real current close-range view" in text_items[0]
    assert "Candidate ID: route-a; frame index: 10; view: front." in text_items
    binary_items = [item for item in agent.content if isinstance(item, BinaryContent)]
    assert [item.media_type for item in binary_items] == ["image/png"]
    assert [item.identifier for item in binary_items] == [
        "route-a_frame_010_front.png",
    ]
    decoded = np.asarray(Image.open(io.BytesIO(binary_items[0].data)).convert("RGB"))
    assert decoded.shape == source_front.shape
    assert np.array_equal(decoded, source_front)


def test_verifier_accepts_direct_string_agent_output():
    class Agent:
        def run_sync(self, _content):
            return (
                '{"keep":["route-a"],"dominant_id":"route-a",'
                '"continue_search":false,"reason":"selected"}'
            )

    verifier = object.__new__(VerifierAgent)
    verifier._agent = Agent()
    decision = verifier.decide(
        depth=1,
        candidates=[candidate("route-a"), candidate("route-b")],
    )

    assert decision == {
        "keep": ["route-a"],
        "dominant_id": "route-a",
        "continue_search": False,
        "reason": "selected",
    }


def test_verifier_normalizes_single_candidate_keep_string():
    class Agent:
        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps({
                "keep": "route-b",
                "dominant_id": "route-b",
                "continue_search": False,
                "reason": "route-b is visually best",
            }))

    verifier = object.__new__(VerifierAgent)
    verifier._agent = Agent()

    decision = verifier.decide(
        depth=1,
        candidates=[candidate("route-a"), candidate("route-b")],
    )

    assert decision == {
        "keep": ["route-b"],
        "dominant_id": "route-b",
        "continue_search": False,
        "reason": "route-b is visually best",
    }


def test_verifier_normalizes_json_encoded_keep_list():
    class Agent:
        def run_sync(self, _content):
            return SimpleNamespace(output=json.dumps({
                "keep": '["route-b", "route-a", "route-b"]',
                "dominant_id": "route-b",
                "continue_search": True,
                "reason": "both branches remain plausible",
            }))

    verifier = object.__new__(VerifierAgent)
    verifier._agent = Agent()

    decision = verifier.decide(
        depth=1,
        candidates=[candidate("route-a"), candidate("route-b")],
    )

    assert decision == {
        "keep": ["route-b", "route-a"],
        "dominant_id": "route-b",
        "continue_search": True,
        "reason": "both branches remain plausible",
    }


def test_agent_runner_retries_responses_raw_string_attribute_error(monkeypatch):
    from parallelworld_manipulation.parallelworld.verifier import _run_agent_output

    class Agent:
        def __init__(self):
            self.calls = 0

        def run_sync(self, _content):
            self.calls += 1
            if self.calls == 1:
                raise AttributeError("'str' object has no attribute 'output'")
            return SimpleNamespace(output="recovered")

    sleeps = []
    monkeypatch.setattr("parallelworld_manipulation.parallelworld.verifier.time.sleep", sleeps.append)
    agent = Agent()

    assert _run_agent_output(agent, "request") == "recovered"
    assert agent.calls == 2
    assert sleeps == [0.25]


def test_agent_runner_keeps_legacy_data_result_compatibility():
    from parallelworld_manipulation.parallelworld.verifier import _run_agent_output

    class Agent:
        def run_sync(self, _content):
            return SimpleNamespace(data="legacy-output")

    assert _run_agent_output(Agent(), "request") == "legacy-output"


def test_openai_responses_agent_accepts_gateway_top_level_string(monkeypatch):
    from pydantic_ai.models.openai import OpenAIResponsesModel

    from parallelworld_manipulation.parallelworld.verifier import _make_agent

    raw_output = '{"candidates":[]}'

    async def raw_string_response(self, *args, **kwargs):
        return raw_output

    monkeypatch.setattr(
        OpenAIResponsesModel, "_responses_create", raw_string_response,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = _make_agent(
        model="openai:gpt-5.5",
        base_url="https://example.invalid/v1",
        system_prompt="Return JSON only.",
    )

    result = agent.run_sync("Return an empty candidate list.")

    assert result.output == raw_output
    response = result.all_messages()[-1]
    assert response.provider_details == {"compat_raw_string_response": True}


def test_world_model_client_accepts_front_only_response():
    class Rpc:
        def call(self, _method, **_kwargs):
            return {"front": np.zeros((1, 2, 3, 4, 3), dtype=np.uint8)}

    client = VideoWorldModelClient(Rpc())
    result = client.rollout(
        front=np.zeros((3, 4, 3), dtype=np.uint8),
        wrist=np.zeros((3, 4, 3), dtype=np.uint8),
        instruction="move", horizon=2, samples=1, stochastic=False, seed=0,
    )

    assert set(result) == {"front"}
    assert result["front"].shape == (1, 2, 3, 4, 3)


def test_wan_rollout_does_not_generate_wrist_view():
    facade = object.__new__(WanLiberoWorldModelFacade)
    calls = []

    def generate_view(image, instruction, horizon, seed):
        calls.append((int(image[0, 0, 0]), instruction, horizon, seed))
        return np.full((2, 3, 4, 3), seed % 255, dtype=np.uint8)

    facade._view = generate_view
    result = facade.rollout(
        front=np.full((3, 4, 3), 7, dtype=np.uint8),
        wrist=np.full((3, 4, 3), 99, dtype=np.uint8),
        instruction="move", horizon=49, samples=2, stochastic=True, seed=10,
    )

    assert set(result) == {"front"}
    assert calls == [(7, "move", 49, 10), (7, "move", 49, 1019)]


def test_wan_uses_endpoint_compatible_negative_prompt():
    class Pipe:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return np.zeros((3, 5, 2, 2), dtype=np.uint8)

    facade = object.__new__(WanLiberoWorldModelFacade)
    facade._pipe = Pipe()
    facade._size = (4, 3)
    facade._max_area = 12
    facade._frame_num = 5
    facade._steps = 1

    facade._view(
        np.zeros((3, 4, 3), dtype=np.uint8),
        "perform one action and hold its endpoint",
        5,
        7,
    )

    assert facade._pipe.kwargs["n_prompt"] == WAN_NEGATIVE_PROMPT
    lowered = WAN_NEGATIVE_PROMPT.lower()
    assert "静态" not in lowered
    assert "静止" not in lowered
    assert "static" not in lowered


def test_minimax_rollout_does_not_generate_wrist_view():
    class Backend:
        def __init__(self):
            self.calls = []

        def generate(self, image, **kwargs):
            self.calls.append((int(image[0, 0, 0]), kwargs))
            return np.zeros((2, 3, 4, 3), dtype=np.uint8)

    facade = object.__new__(MiniMaxH3WorldModelFacade)
    facade._backend = Backend()
    result = facade.rollout(
        front=np.full((3, 4, 3), 7, dtype=np.uint8),
        wrist=np.full((3, 4, 3), 99, dtype=np.uint8),
        instruction="move", horizon=22, samples=2, stochastic=True, seed=10,
    )

    assert set(result) == {"front"}
    assert [call[0] for call in facade._backend.calls] == [7, 7]
    assert [call[1]["seed"] for call in facade._backend.calls] == [10, 1019]


def test_search_expands_retained_route_with_planner_callback():
    class Verifier:
        def decide(self, *, depth, **_kwargs):
            if depth == 1:
                return {
                    "keep": ["root"], "dominant_id": None,
                    "continue_search": True, "reason": "expand",
                }
            return {
                "keep": ["root::child-0"], "dominant_id": "root::child-0",
                "continue_search": False, "reason": "selected",
            }

    client = FakeClient()
    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=client, tts=False,
        tts_samples=3, rollout_steps=1, verifier=Verifier(),
    )

    expanded_from = []

    def expand(route, _limit):
        expanded_from.append(route)
        return [candidate(f"child-{index}") for index in range(4)]

    result = explorer.search([candidate("root")], next_candidate_generator=expand)

    assert result["_selected_candidate"]["id"] == "root::child-0"
    assert len(explorer._last_search_history) == 2
    assert len(client.calls) == 5
    assert np.array_equal(expanded_from[0]["_imagined_front"], client.calls[0]["front"])
    assert "_imagined_wrist" not in expanded_from[0]


def test_children_continue_from_their_own_parent_imagined_states():
    class LineageClient(FakeClient):
        def rollout(self, **kwargs):
            self.calls.append(kwargs)
            value = len(self.calls)
            front = np.clip(kwargs["front"].astype(np.int16) + value, 0, 255).astype(np.uint8)
            return {"front": front[None, None]}

    class Verifier:
        def decide(self, *, depth, **_kwargs):
            if depth == 1:
                return {
                    "keep": ["a", "b"], "dominant_id": "a",
                    "continue_search": True, "reason": "expand both",
                }
            return {
                "keep": ["a::next"], "dominant_id": "a::next",
                "continue_search": False, "reason": "selected",
            }

    class Enhancer:
        def __init__(self):
            self.calls = []

        def enhance_batch(self, *, candidates, task, fronts, rollout_steps):
            self.calls.append({
                "ids": [item["id"] for item in candidates],
                "fronts": {key: value.copy() for key, value in fronts.items()},
                "task": task,
                "rollout_steps": rollout_steps,
            })
            return {
                item["id"]: {
                    "instruction": "The robot performs this action and holds the final state.",
                    "source": "test_enhancer",
                }
                for item in candidates
            }

    client = LineageClient()
    enhancer = Enhancer()
    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=client, tts=False,
        tts_samples=3, rollout_steps=1, verifier=Verifier(),
        prompt_enhancer=enhancer,
    )
    planner_inputs = {}

    def expand(route, _limit):
        planner_inputs[route["id"]] = route["_imagined_front"].copy()
        return [candidate("next")]

    result = explorer.search(
        [candidate("a"), candidate("b")], next_candidate_generator=expand
    )

    assert result["_selected_candidate"]["id"] == "a::next"
    assert np.all(planner_inputs["a"] == 1)
    assert np.all(planner_inputs["b"] == 2)
    assert np.all(client.calls[2]["front"] == 1)
    assert np.all(client.calls[3]["front"] == 2)
    assert len(enhancer.calls) == 2
    assert enhancer.calls[1]["ids"] == ["a::next", "b::next"]
    assert np.all(enhancer.calls[1]["fronts"]["a::next"] == 1)
    assert np.all(enhancer.calls[1]["fronts"]["b::next"] == 2)
    assert explorer._last_rollout_videos[0]["parent_id"] == "a"
    assert explorer._last_rollout_videos[1]["parent_id"] == "b"


def test_deeper_candidate_does_not_fall_back_when_parent_state_is_missing():
    client = FakeClient()
    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=client, tts=False,
        tts_samples=3, rollout_steps=1,
    )

    result = explorer.decide(
        candidates=[{**candidate("orphan"), "parent_id": "missing"}], depth=2
    )

    assert "unavailable" in result["branches"][0]["world_model_error"]
    assert client.calls == []


def test_world_model_is_opt_in_and_tts_has_explicit_switch():
    assert _parallelworld_runtime(SimpleNamespace(world_model_endpoint=None)) == {}
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=False)

    enabled = parser.parse_args(
        ["--suite", "libero_spatial_task", "--task", "0", "--wm-tts"]
    )
    disabled = parser.parse_args(
        ["--suite", "libero_spatial_task", "--task", "0", "--no-wm-tts"]
    )
    assert enabled.wm_tts is True
    assert disabled.wm_tts is False


@pytest.mark.parametrize("generated_wrist", [False, True])
def test_wrist_artifacts_require_a_generated_view(generated_wrist):
    class Client(FakeClient):
        def rollout(self, **kwargs):
            result = super().rollout(**kwargs)
            if generated_wrist:
                result["wrist"] = kwargs["wrist"][None, None]
            return result

    explorer = ParallelWorldExplorer(
        primitives=primitives(), client=Client(), tts=False,
        tts_samples=1, rollout_steps=1,
    )
    routes = [candidate("route-a")]
    decision = explorer.search(routes)
    saved = {}
    toolkit = object.__new__(LiberoToolkit)
    toolkit._parallelworld_explorer = explorer
    toolkit._parallelworld_trace = []
    toolkit._parallelworld_timing_summary = []
    toolkit._state = SimpleNamespace(save=lambda filename, data, **kwargs: saved.update({filename: data}))
    toolkit._append_parallelworld_index = lambda **kwargs: None
    toolkit._save_parallelworld_viewer = lambda: None
    toolkit._save_parallelworld_artifacts(
        decision, action_step=1, action_name="release", requested_candidates=routes,
    )
    assert any(name.endswith("input_real_wrist.png") for name in saved)
    assert any("result__" in name and name.endswith("__wrist.png") for name in saved) == generated_wrist
    assert any("world_model_preview__" in name and name.endswith("__wrist.mp4") for name in saved) == generated_wrist
    action_record = next(data for name, data in saved.items() if name.endswith("/action.json"))
    assert ("final_wrist" in action_record["artifacts"]) == generated_wrist


def test_physical_tool_is_routed_through_parallelworld(monkeypatch):
    toolkit = object.__new__(LiberoToolkit)
    toolkit._parallelworld_gate = True
    toolkit._parallelworld_select = lambda name, values: (
        "release", {"selected": values["candidate_routes"][0]["id"]},
    )
    seen = {}

    def fake_execute(_self, name, input_dict):
        seen.update(name=name, input_dict=input_dict)
        return ToolResult(name=name, result={"ok": True})

    monkeypatch.setattr(Toolkit, "execute_tool", fake_execute)
    result = toolkit.execute_tool(
        "release", {"candidate_routes": [{"id": "small"}]}
    )

    assert result.result == {"ok": True}
    assert seen == {"name": "release", "input_dict": {"selected": "small"}}


def test_single_planner_route_bypasses_parallelworld(monkeypatch):
    toolkit = object.__new__(LiberoToolkit)

    class ExplodingExplorer:
        N_MAX = 4

        def search(self, **_kwargs):
            raise AssertionError("single route must not enter ParallelWorld")

    toolkit._parallelworld_explorer = ExplodingExplorer()
    toolkit._parallelworld_allowed_primitives = {"release"}
    toolkit._pending_parallel_execution = {"stale": True}
    selected_name, selected_args = toolkit._parallelworld_select(
        "parallelworld_decide",
        {
            "candidate_routes": [
                {
                    "id": "only-route",
                    "primitive": "release",
                    "arguments": {"gripper": -1},
                    "visual_action": {"subject": "gripper", "motion": "opens"},
                }
            ]
        },
    )
    assert selected_name == "release"
    assert selected_args == {"gripper": -1}
    assert toolkit._pending_parallel_execution is None
