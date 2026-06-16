"""
run_libero_eval.py

Evaluates OpenVLA in the LIBERO simulation environment with comprehensive metrics:
  - Task success rate
  - Per-episode step count
  - Per-step latency (total, model forward, postprocess)
  - GPU memory usage
  - Visual token count (if available)
  - Episode total time
  - Rollout videos

Output: outputs/{run_id}/summary.json + episode_metrics.csv

Usage:
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint /infini-data/checkpoints/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 1 \
        --use_wandb False
"""

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_action_timed,
    get_gpu_memory_mb,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    reset_gpu_memory_stats,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    #################################################################################################################
    # Metrics & Logging
    #################################################################################################################
    save_dir: str = "./outputs"
    save_video: bool = True
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7
    # fmt: on


MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must be set!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    # --- Output directory ---
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    output_dir = os.path.join(cfg.save_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Logging ---
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    # --- Load model ---
    print("Loading model...")
    model, processor = get_model(cfg)

    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Unnorm key {cfg.unnorm_key} not in norm_stats!"

    # --- WandB ---
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # --- Task suite ---
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name} ({num_tasks_in_suite} tasks)")

    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS_MAP.get(cfg.task_suite_name, 400)

    # --- Metrics accumulators ---
    all_episodes = []
    total_episodes, total_successes = 0, 0
    total_forward_ms, total_preprocess_ms, total_postprocess_ms, total_step_ms = 0.0, 0.0, 0.0, 0.0
    total_inference_steps = 0

    # Per-task accumulators
    task_successes_dict = {}
    task_forward_ms_dict = {}
    task_postprocess_ms_dict = {}
    task_steps_dict = {}

    reset_gpu_memory_stats()
    max_gpu_mem = 0.0

    # --- Reset memory stats after model load (exclude model loading from eval stats) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        task_name = task_description[:60].replace(" ", "_").replace("\n", "_")

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc="  Trial", leave=False):
            print(f"\n[Task {task_id+1}/{num_tasks_in_suite}] {task_description}")
            log_file.write(f"\n[Task {task_id+1}/{num_tasks_in_suite}] {task_description}\n")
            log_file.flush()

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            episode_start = time.perf_counter()

            # Per-episode metrics
            ep_forward_ms, ep_preprocess_ms, ep_postprocess_ms, ep_step_ms = [], [], [], []
            ep_gpu_mem = 0.0

            print(f"  Starting episode {episode_idx+1}...")

            while t < max_steps + cfg.num_steps_wait:
                step_start = time.perf_counter()

                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    step_end = time.perf_counter()
                    ep_step_ms.append((step_end - step_start) * 1000.0)
                    continue

                try:
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # --- Timed inference ---
                    action, step_metrics = get_action_timed(cfg, model, observation, task_description, processor=processor)

                    # --- Timed action postprocessing ---
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    postprocess_start = time.perf_counter()
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    postprocess_time_ms = (time.perf_counter() - postprocess_start) * 1000.0

                    # --- Environment step ---
                    obs, reward, done, info = env.step(action.tolist())

                    step_end = time.perf_counter()
                    total_ms = (step_end - step_start) * 1000.0
                    ep_step_ms.append(total_ms)
                    ep_forward_ms.append(step_metrics["model_forward_ms"])
                    ep_preprocess_ms.append(step_metrics["preprocess_ms"])
                    ep_postprocess_ms.append(postprocess_time_ms)

                    current_gpu_mem = get_gpu_memory_mb()
                    if current_gpu_mem is not None and current_gpu_mem > ep_gpu_mem:
                        ep_gpu_mem = current_gpu_mem

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

            episode_end = time.perf_counter()
            episode_time = episode_end - episode_start
            inference_steps = len(ep_forward_ms)

            # Skip settling steps for averages
            step_ms_valid = ep_step_ms[cfg.num_steps_wait:] if len(ep_step_ms) > cfg.num_steps_wait else ep_step_ms

            avg_forward = sum(ep_forward_ms) / len(ep_forward_ms) if ep_forward_ms else 0.0
            avg_preprocess = sum(ep_preprocess_ms) / len(ep_preprocess_ms) if ep_preprocess_ms else 0.0
            avg_postprocess = sum(ep_postprocess_ms) / len(ep_postprocess_ms) if ep_postprocess_ms else 0.0
            avg_step = sum(step_ms_valid) / len(step_ms_valid) if step_ms_valid else 0.0

            total_forward_ms += sum(ep_forward_ms)
            total_preprocess_ms += sum(ep_preprocess_ms)
            total_postprocess_ms += sum(ep_postprocess_ms)
            total_step_ms += sum(step_ms_valid)
            total_inference_steps += inference_steps

            if ep_gpu_mem > max_gpu_mem:
                max_gpu_mem = ep_gpu_mem

            episode_record = {
                "task_name": task_name,
                "task_description": task_description,
                "episode_id": episode_idx,
                "success": done,
                "num_steps": t,
                "num_inference_steps": inference_steps,
                "episode_time_sec": round(episode_time, 4),
                "avg_step_time_ms": round(avg_step, 3),
                "avg_model_forward_ms": round(avg_forward, 3),
                "std_model_forward_ms": round(np.std(ep_forward_ms), 3) if ep_forward_ms else 0.0,
                "avg_preprocess_ms": round(avg_preprocess, 3),
                "avg_action_postprocess_ms": round(avg_postprocess, 3),
                "max_gpu_memory_mb": round(ep_gpu_mem, 1) if ep_gpu_mem else 0.0,
                "seed": cfg.seed,
            }
            all_episodes.append(episode_record)

            task_episodes += 1
            total_episodes += 1

            # Save video
            if cfg.save_video:
                save_rollout_video(
                    replay_images, total_episodes, success=done,
                    task_description=task_description, log_file=log_file,
                    output_dir=output_dir,
                )

            success_pct = total_successes / total_episodes * 100
            print(f"  Success: {done} | Total: {total_successes}/{total_episodes} ({success_pct:.1f}%)")
            print(f"    Avg forward: {avg_forward:.1f}ms | Avg step: {avg_step:.1f}ms | GPU mem: {ep_gpu_mem:.0f}MB")

            log_file.write(f"Success: {done}\n")
            log_file.write(f"Total: {total_successes}/{total_episodes} ({success_pct:.1f}%)\n")
            log_file.flush()

        env.close()
        print(f"  Task {task_id+1} success rate: {task_successes}/{task_episodes}")

    log_file.close()

    # --- Write metrics CSV ---
    csv_path = os.path.join(output_dir, "episode_metrics.csv")
    if all_episodes:
        fieldnames = list(all_episodes[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episodes)
        print(f"Episode metrics saved to: {csv_path}")

    # --- Write summary JSON ---
    overall_success = total_successes / total_episodes if total_episodes else 0.0
    overall_forward = total_forward_ms / total_inference_steps if total_inference_steps else 0.0
    overall_preprocess = total_preprocess_ms / total_inference_steps if total_inference_steps else 0.0
    overall_postprocess = total_postprocess_ms / total_inference_steps if total_inference_steps else 0.0
    overall_step = total_step_ms / total_inference_steps if total_inference_steps else 0.0

    # Per-task aggregation
    per_task = {}
    task_names_seen = set()
    for ep in all_episodes:
        tn = ep["task_name"]
        if tn not in per_task:
            per_task[tn] = {"successes": 0, "total": 0, "forward_ms": [], "preprocess_ms": [], "latency_ms": [], "steps": []}
        per_task[tn]["successes"] += int(ep["success"])
        per_task[tn]["total"] += 1
        per_task[tn]["forward_ms"].append(ep["avg_model_forward_ms"])
        per_task[tn]["preprocess_ms"].append(ep["avg_preprocess_ms"])
        per_task[tn]["latency_ms"].append(ep["avg_step_time_ms"])
        per_task[tn]["steps"].append(ep["num_inference_steps"])

    per_task_summary = {}
    for tn, data in per_task.items():
        sr = data["successes"] / data["total"] if data["total"] else 0.0
        per_task_summary[tn] = {
            "success_rate": round(sr, 3),
            "num_episodes": data["total"],
            "mean_forward_ms": round(sum(data["forward_ms"]) / len(data["forward_ms"]), 3) if data["forward_ms"] else 0.0,
            "mean_preprocess_ms": round(sum(data["preprocess_ms"]) / len(data["preprocess_ms"]), 3) if data["preprocess_ms"] else 0.0,
            "mean_latency_ms": round(sum(data["latency_ms"]) / len(data["latency_ms"]), 3) if data["latency_ms"] else 0.0,
            "mean_steps": round(sum(data["steps"]) / len(data["steps"]), 1) if data["steps"] else 0.0,
        }

    summary = {
        "model": cfg.model_family,
        "checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite": cfg.task_suite_name,
        "num_tasks_evaluated": num_tasks_in_suite,
        "num_trials_per_task": cfg.num_trials_per_task,
        "num_episodes": total_episodes,
        "overall_success_rate": round(overall_success, 4),
        "mean_episode_steps": round(sum(ep["num_inference_steps"] for ep in all_episodes) / total_episodes, 1) if total_episodes else 0,
        "mean_episode_time_sec": round(sum(ep["episode_time_sec"] for ep in all_episodes) / total_episodes, 4) if total_episodes else 0,
        "mean_step_time_ms": round(overall_step, 3),
        "mean_model_forward_time_ms": round(overall_forward, 3),
        "std_model_forward_time_ms": round(np.std([ep["avg_model_forward_ms"] for ep in all_episodes]), 3) if all_episodes else 0,
        "mean_preprocess_time_ms": round(overall_preprocess, 3),
        "mean_action_postprocess_time_ms": round(overall_postprocess, 3),
        "max_gpu_memory_mb": round(max_gpu_mem, 1),
        "per_task": per_task_summary,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # --- WandB ---
    if cfg.use_wandb:
        wandb.log({
            "overall_success_rate": overall_success,
            "mean_model_forward_ms": overall_forward,
            "mean_step_time_ms": overall_step,
            "max_gpu_memory_mb": max_gpu_mem,
        })
        wandb.save(csv_path)
        wandb.save(summary_path)
        wandb.save(local_log_filepath)

    print(f"\n=== DONE ===")
    print(f"Tasks evaluated: {num_tasks_in_suite}")
    print(f"Success rate: {total_successes}/{total_episodes} ({overall_success*100:.1f}%)")
    print(f"Mean model forward: {overall_forward:.1f}ms | Mean step: {overall_step:.1f}ms")
    print(f"Max GPU memory: {max_gpu_mem:.0f}MB")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    eval_libero()
