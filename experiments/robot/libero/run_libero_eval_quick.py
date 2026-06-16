"""Quick evaluation: 10 tasks, 1 trial each."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1      # 1 trial per task for quick eval
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must be set!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    print("Loading model...")
    model, processor = get_model(cfg)

    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    # Limit to first 10 tasks
    max_tasks = 10
    num_tasks = min(max_tasks, num_tasks_in_suite)
    print(f"Task suite: {cfg.task_suite_name} ({num_tasks} tasks, {cfg.num_trials_per_task} trial(s) each)")

    resize_size = get_image_resize_size(cfg)

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(cfg.task_suite_name, 400)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"  Trial", leave=False):
            print(f"\n[Task {task_id+1}/{num_tasks}] {task_description}")
            log_file.write(f"\n[Task {task_id+1}/{num_tasks}] {task_description}\n")
            log_file.flush()

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            print(f"  Starting episode {episode_idx+1}...")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    action = get_action(cfg, model, observation, task_description, processor=processor)
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"  Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    log_file.flush()
                    break

            task_episodes += 1
            total_episodes += 1

            success = done
            print(f"  Success: {success} | Total: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"Success: {success}\n")
            log_file.write(f"Total: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)\n")
            log_file.flush()

        env.close()
        print(f"  Task {task_id+1} success rate: {task_successes}/{task_episodes}")

    log_file.write(f"\n=== FINAL RESULTS ===\n")
    log_file.write(f"Tasks evaluated: {num_tasks}\n")
    log_file.write(f"Total success rate: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)\n")
    log_file.close()

    print(f"\n=== DONE ===")
    print(f"Tasks evaluated: {num_tasks}")
    print(f"Total success rate: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.1f}%)")
    print(f"Log: {local_log_filepath}")


if __name__ == "__main__":
    eval_libero()
