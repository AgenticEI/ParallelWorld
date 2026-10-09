"""LIBERO tool schema for optional Agentic Planner imagination."""

PARALLELWORLD_TOOL_SPEC = {
    "name": "parallelworld_decide",
    "description": (
        "Propose several genuinely different candidate routes for the next physical "
        "step. Candidates may and should mix different primitives as well as parameter "
        "choices; do not produce only parameter variants of one primitive. Do not provide "
        "a model prompt. For each candidate, provide visual_action with the grounded "
        "subject, target object and distinguishing qualifier, relative motion, gripper "
        "state, contact behavior, and visible outcome. Keep simulator coordinates and "
        "control parameters only in arguments; never copy them into visual_action, "
        "except for the optional normalized target_uv image coordinate. When the "
        "target center has been localized in the current agentview image, provide "
        "target_uv so the runtime can mark it consistently for the world model. The "
        "runtime converts visual_action into a natural-language video-model I2V prompt, "
        "previews all candidates, and lets the Verifier select one. "
        "Observation and state-reading tools bypass this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "depth": {"type": "integer", "minimum": 1, "default": 1},
            "candidate_routes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "primitive": {
                            "type": "string",
                            "enum": [
                                "pi0_pick",
                                "pi0_doubled",
                                "move_to",
                                "move_pose",
                                "navigate_to",
                                "move_base",
                                "release",
                                "set_gripper",
                                "rotate_wrist",
                                "rotate_pitch",
                            ],
                        },
                        "arguments": {"type": "object"},
                        "rationale": {"type": "string"},
                        "visual_action": {
                            "type": "object",
                            "description": (
                                "Grounded, purely visual description of this candidate. "
                                "Do not include simulator coordinates, primitive/API names, "
                                "numeric commands, tolerances, step sizes, counts, or "
                                "thresholds. target_uv is the only allowed numeric field."
                            ),
                            "properties": {
                                "subject": {
                                    "type": "string",
                                    "description": "Visible actor, e.g. robot arm or gripper.",
                                },
                                "motion": {
                                    "type": "string",
                                    "description": (
                                        "Third-person visual verb phrase, e.g. moves smoothly."
                                    ),
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Grounded target object name.",
                                },
                                "target_qualifier": {
                                    "type": "string",
                                    "description": (
                                        "Visual or relational instance qualifier, e.g. "
                                        "the far-right one that is not between the plate "
                                        "and ramekin."
                                    ),
                                },
                                "target_uv": {
                                    "type": "array",
                                    "items": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": (
                                        "Optional target center [u, v] in the currently "
                                        "displayed agentview calibration image, normalized "
                                        "to [0,1], with origin top-left, +u right, +v down. "
                                        "Supply only when the target center was visually "
                                        "localized in that exact image."
                                    ),
                                },
                                "spatial_relation": {
                                    "type": "string",
                                    "description": "Relative relation such as above or toward.",
                                },
                                "gripper_state": {
                                    "type": "string",
                                    "enum": ["open", "closed", "holding", "unchanged"],
                                },
                                "contact": {
                                    "type": "string",
                                    "description": (
                                        "Observable contact behavior in ordinary language."
                                    ),
                                },
                                "intended_outcome": {
                                    "type": "string",
                                    "description": "Expected visible result of this motion.",
                                },
                            },
                            "required": ["subject", "motion"],
                        },
                    },
                    "required": ["id", "primitive", "arguments", "visual_action"],
                },
            },
        },
        "required": ["candidate_routes"],
    },
}
