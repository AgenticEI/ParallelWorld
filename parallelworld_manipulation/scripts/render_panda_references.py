#!/usr/bin/env python3
"""Render clean Panda / PandaGripper identity references with robosuite."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    back = camera - target
    back /= np.linalg.norm(back)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.column_stack([right, up, back])


def _set_camera(model, name: str, position: np.ndarray, rotation: np.ndarray) -> None:
    camera_id = model.camera_name2id(name)
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = _wxyz(rotation)


def _clean_scene(model) -> None:
    """Keep the exact robot, gripper, mount, and neutral room only."""
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id) or ""
        keep = (
            name == "floor"
            or name.startswith("wall_")
            or name.startswith("robot0_")
            or name.startswith("gripper0_")
        )
        if not keep:
            model.geom_rgba[geom_id, 3] = 0.0
        elif name == "floor" or name.startswith("wall_"):
            model.geom_rgba[geom_id] = [0.36, 0.36, 0.36, 1.0]


def _render(sim, camera: str, size: int) -> Image.Image:
    rgb = sim.render(camera_name=camera, height=size, width=size)
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8)[::-1]).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    env = suite.make(
        "Lift",
        robots="Panda",
        gripper_types="default",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    try:
        env.reset()
        model = env.sim.model
        _clean_scene(model)

        # A three-quarter view is deliberate: from the task's front agentview
        # the folded Panda links overlap almost perfectly, which makes a poor
        # identity reference even though it is faithful to the task camera.
        full_position = np.asarray([-0.05, 1.18, 1.40])
        full_target = np.asarray([-0.34, 0.0, 1.23])
        _set_camera(model, "agentview", full_position, _look_at(full_position, full_target))
        env.sim.forward()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        _render(env.sim, "agentview", args.size).save(
            args.output_dir / "panda_full_reference.png"
        )

        eef_site = model.site_name2id("gripper0_right_grip_site")
        eef = env.sim.data.site_xpos[eef_site].copy()
        close_position = eef + np.asarray([0.34, 0.30, 0.16])
        close_target = eef + np.asarray([0.0, 0.0, 0.03])
        _set_camera(
            model,
            "agentview",
            close_position,
            _look_at(close_position, close_target),
        )
        env.sim.forward()
        _render(env.sim, "agentview", args.size).save(
            args.output_dir / "panda_gripper_reference.png"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
