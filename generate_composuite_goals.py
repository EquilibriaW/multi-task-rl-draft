"""Utility to generate presampled goal banks for CompoSuite experiments.

Run:
  python generate_composuite_goals.py --output_dir=data/composuite_goals \
         --camera_name=frontview --img_size=64

This will create two pickled dictionaries:
    <output_dir>/train_goals.pkl
    <output_dir>/test_goals.pkl

Each dictionary maps the string "<Robot><EnvName>" (the same `.name` field
exposed by `composuite.env.gym_wrapper.GymWrapper`) to a dict with keys:
    "state": flattened state-vector goal (np.float32)
    "image": flattened RGB goal image (np.uint8)

These files are required by `env_utils.CompoSuite[State|Image]Wrapper`.
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict

import numpy as np
import tqdm

import composuite
from composuite.env.gym_wrapper import GymWrapper as CSGymWrapper


def hide_goal_geoms(env):
  """Makes goal geoms fully transparent so they don't appear in the rendered image."""
  geom_names = env.sim.model.geom_names
  rgba = env.sim.model.geom_rgba
  for i, name in enumerate(geom_names):
    # MuJoCo may store names as bytes; make robust.
    name_str = name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else str(name)
    if "goal" in name_str.lower():
      rgba[i, 3] = 0.0  # alpha channel


def make_env(task, use_camera_obs: bool, camera_names):
  robot, obj, obstacle, objective = task
  env = composuite.env.main.make(
      robot=robot,
      obj=obj,
      obstacle=obstacle,
      task=objective,
      controller="joint",
      env_horizon=500,
      has_renderer=False,
      has_offscreen_renderer=use_camera_obs,
      use_camera_obs=use_camera_obs,
      camera_names=camera_names,
      reward_shaping=False,
  )
  hide_goal_geoms(env)
  return env


def flatten_state(obs) -> np.ndarray:
  """Return a 1-D float32 state vector from dict or already-flat array."""
  if isinstance(obs, dict):
    return np.concatenate([obs[k].ravel() for k in sorted(obs.keys())]).astype(np.float32)
  # Already a flat (or nested) numpy array.
  return np.asarray(obs, dtype=np.float32).ravel()


def capture_goal(env, camera_name: str, img_size: int) -> np.ndarray:
  img = env.sim.render(width=img_size, height=img_size, camera_name=camera_name)
  if img.dtype != np.uint8:
    img = (255 * img).astype(np.uint8)
  return img.flatten()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", type=str, default="data/composuite_goals",
                      help="Directory to store pickled goal files.")
  parser.add_argument("--img_size", type=int, default=64,
                      help="Side length of square RGB goal thumbnails.")
  parser.add_argument("--camera_name", type=str, default="frontview",
                      help="Camera name to render goal images from.")
  parser.add_argument("--num_train", type=int, default=256)
  parser.add_argument("--num_test", type=int, default=256)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  train_tasks, test_tasks = composuite.sample_tasks(
      experiment_type="default", num_train=args.num_train)
  splits = {"train": train_tasks, "test": test_tasks[:args.num_test]}

  for split, tasks in splits.items():
    goal_bank = {}
    for task in tqdm.tqdm(tasks, desc=f"{split} tasks"):
      # goal image ----------------------------------------------------------------
      env_img = make_env(task, use_camera_obs=True, camera_names=[args.camera_name])
      env_img.reset()  # object already at goal position after reset
      goal_image = capture_goal(env_img, args.camera_name, args.img_size)
      env_img.close()

      # goal state ----------------------------------------------------------------
      env_state = make_env(task, use_camera_obs=False, camera_names=[args.camera_name])
      obs_dict = env_state.reset()
      flat_state = flatten_state(obs_dict)
      env_state.close()

      task_key = f"{task[0]}_{task[1]}_{task[2]}_{task[3]}"

      goal_bank[task_key] = {
          "state": flat_state,
          "image": goal_image,
      }

    out_path = os.path.join(args.output_dir, f"{split}_goals.pkl")
    with open(out_path, "wb") as f:
      pickle.dump(goal_bank, f)
    print(f"Wrote {len(goal_bank)} goals to {out_path}")


if __name__ == "__main__":
  main()
