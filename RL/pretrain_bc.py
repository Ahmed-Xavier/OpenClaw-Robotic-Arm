"""pretrain_bc.py — Behaviour-clone the expert into a SB3 MlpPolicy warm-start.

Pipeline:
  1. Collect N expert rollouts (seeds 0..N-1) via expert.expert_action().
  2. Store (obs, action) pairs in memory.
  3. Train a PyTorch MLP on MSE(pred_action, expert_action) for `epochs` epochs.
  4. Load the MLP weights into a fresh SB3 PPO MlpPolicy.
  5. Save the PPO object to `save_path` so train.py can load it with --init-from.

The saved file is a normal SB3 PPO .zip — you can evaluate it with evaluate.py
or use it as the starting point for PPO fine-tuning via train.py --init-from.

Usage
-----
  python RL/pretrain_bc.py [--rollouts 200] [--epochs 50] [--batch-size 512]
                            [--lr 3e-4] [--save-path RL/models/bc_grasp_ppo]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from RL.envs.grasp_env import GraspEnv
from RL.expert import expert_action


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ROLLOUTS: int = 200
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 512
DEFAULT_LR: float = 3e-4
DEFAULT_SAVE_PATH: str = "RL/models/bc_grasp_ppo"


# ---------------------------------------------------------------------------
# Step 1 — Data collection
# ---------------------------------------------------------------------------

def collect_expert_data(
    n_rollouts: int = DEFAULT_ROLLOUTS,
    max_steps: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect expert observations and actions from GraspEnv.

    Args:
        n_rollouts: Number of episodes to collect from (seeds 0..n_rollouts-1).
        max_steps: Max episode length (matches GraspEnv.max_episode_steps).

    Returns:
        (obs_array, act_array): shape (N, 10) and (N, 4) float32 arrays.
    """
    print(f"[BC] Collecting expert data: {n_rollouts} rollouts × {max_steps} steps max...")
    env = GraspEnv(has_renderer=False)

    obs_list = []
    act_list = []
    n_success = 0

    t_start = time.time()
    for seed in range(n_rollouts):
        obs, _ = env.reset(seed=seed)
        ep_obs = []
        ep_act = []
        for _ in range(max_steps):
            act = expert_action(obs)
            ep_obs.append(obs.copy())
            ep_act.append(act.copy())
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        if info.get("is_grasped", False):
            n_success += 1
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)

        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t_start
            sr = n_success / (seed + 1) * 100
            print(f"  [BC] {seed+1}/{n_rollouts} rollouts | expert SR so far: "
                  f"{n_success}/{seed+1} ({sr:.1f}%) | {elapsed:.1f}s")

    env.close()
    if len(obs_list) == 0:
        raise RuntimeError("No successful rollouts collected by expert!")
    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
    act_arr = np.stack(act_list, axis=0).astype(np.float32)
    sr = n_success / n_rollouts * 100
    print(f"[BC] Kept {len(obs_arr)} transitions from {n_success}/{n_rollouts} successful episodes "
          f"(expert SR: {sr:.1f}%)")
    return obs_arr, act_arr


# ---------------------------------------------------------------------------
# Step 2 — BC training
# ---------------------------------------------------------------------------

def train_bc(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "auto",
) -> nn.Module:
    """Train an MLP on MSE(action, expert_action).

    Returns the trained MLP (same architecture as SB3 MlpPolicy default net).
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[BC] Training on device: {dev}")

    obs_t = torch.tensor(obs_arr, dtype=torch.float32)
    act_t = torch.tensor(act_arr, dtype=torch.float32)

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MLP matching SB3 MlpPolicy default: [64, 64] with tanh, output in [-1, 1]
    mlp = nn.Sequential(
        nn.Linear(10, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 4),  nn.Tanh(),
    ).to(dev)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(dev), act_b.to(dev)
            pred = mlp(obs_b)
            loss = loss_fn(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(obs_b)
        avg_loss = epoch_loss / len(obs_arr)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [BC] Epoch {epoch:3d}/{epochs} | MSE loss: {avg_loss:.6f}")

    return mlp


# ---------------------------------------------------------------------------
# Step 3 — Transfer weights into SB3 PPO MlpPolicy
# ---------------------------------------------------------------------------

def transfer_bc_to_ppo(
    mlp: nn.Module,
    save_path: str = DEFAULT_SAVE_PATH,
    device: str = "auto",
) -> PPO:
    """Create an SB3 PPO object, inject BC weights into the actor net, and save.

    SB3 MlpPolicy actor network layout (default net_arch=[64, 64]):
      policy_net: [Linear(obs_dim→64), activation, Linear(64→64), activation]
      action_net: Linear(64→action_dim)

    The BC MLP layout: Linear(10→64) Tanh Linear(64→64) Tanh Linear(64→4) Tanh
    We map: layers[0]→policy.mlp_extractor.policy_net[0],
            layers[2]→policy.mlp_extractor.policy_net[2],
            layers[4]→policy.action_net
    """
    if device == "auto":
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        torch_device = device

    print("[BC] Creating SB3 PPO to receive BC weights...")
    env = Monitor(GraspEnv(has_renderer=False))
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        verbose=0,
        seed=42,
        device=torch_device,
    )

    # Extract BC MLP weights (on CPU for assignment)
    bc_state = {k: v.cpu() for k, v in mlp.state_dict().items()}

    # Transfer into SB3 policy
    # SB3 policy layers:
    #   mlp_extractor.policy_net.0 -> weight/bias (10→64)
    #   mlp_extractor.policy_net.2 -> weight/bias (64→64)
    #   action_net -> weight/bias (64→4)
    policy = model.policy

    with torch.no_grad():
        # policy_net layer 0 (Linear 10→64)
        policy.mlp_extractor.policy_net[0].weight.copy_(bc_state["0.weight"])
        policy.mlp_extractor.policy_net[0].bias.copy_(bc_state["0.bias"])
        # policy_net layer 2 (Linear 64→64)
        policy.mlp_extractor.policy_net[2].weight.copy_(bc_state["2.weight"])
        policy.mlp_extractor.policy_net[2].bias.copy_(bc_state["2.bias"])
        # action_net (Linear 64→4) — bc has tanh after, SB3 uses separate log_std
        policy.action_net.weight.copy_(bc_state["4.weight"])
        policy.action_net.bias.copy_(bc_state["4.bias"])

    print("[BC] Weights transferred successfully.")

    # Save
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"[BC] Saved BC-initialised PPO to: {save_path}.zip")

    env.close()
    return model


# ---------------------------------------------------------------------------
# Step 4 — Quick BC policy evaluation
# ---------------------------------------------------------------------------

def evaluate_bc(model: PPO, episodes: int = 10, seed_start: int = 0) -> dict:
    """Evaluate BC policy deterministically over N episodes."""
    env = GraspEnv(has_renderer=False)
    successes = 0
    n_close_cmds = []
    dists = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        n_close = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if action[3] > 0:
                n_close += 1
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        successes += int(info.get("is_grasped", False))
        n_close_cmds.append(n_close)
        dists.append(info["distance"])
        status = "SUCCESS" if info.get("is_grasped", False) else "FAIL"
        print(f"  [BC Eval] Ep {ep+1:3d} | {status} | dist={info['distance']*100:.2f}cm | close_cmds={n_close}")

    env.close()
    sr = successes / episodes * 100
    result = {
        "success_rate": sr,
        "successes": successes,
        "episodes": episodes,
        "avg_dist_cm": float(np.mean(dists)) * 100,
        "avg_close_cmds": float(np.mean(n_close_cmds)),
    }
    print(f"  [BC Eval] SR={successes}/{episodes} ({sr:.1f}%) | "
          f"Avg dist={result['avg_dist_cm']:.2f}cm | "
          f"Avg close cmds={result['avg_close_cmds']:.1f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pretrain_bc(
    n_rollouts: int = DEFAULT_ROLLOUTS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    save_path: str = DEFAULT_SAVE_PATH,
    device: str = "auto",
    eval_episodes: int = 10,
) -> PPO:
    print("=" * 65)
    print("  BEHAVIOUR CLONING PRE-TRAINING (Phase 2 GraspEnv)")
    print("=" * 65)
    print(f"Rollouts:   {n_rollouts}")
    print(f"Epochs:     {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR:         {lr}")
    print(f"Device:     {device} (CUDA: {torch.cuda.is_available()})")
    print(f"Save path:  {save_path}")
    print("-" * 65)

    # 1. Collect data
    obs_arr, act_arr = collect_expert_data(n_rollouts=n_rollouts)

    # 2. Train BC MLP
    print(f"\n[BC] Training MLP for {epochs} epochs...")
    mlp = train_bc(obs_arr, act_arr, epochs=epochs, batch_size=batch_size,
                   lr=lr, device=device)

    # 3. Transfer into SB3 PPO
    model = transfer_bc_to_ppo(mlp, save_path=save_path, device=device)

    # 4. Evaluate
    print(f"\n[BC] Evaluating BC policy over {eval_episodes} episodes...")
    evaluate_bc(model, episodes=eval_episodes)

    print("\n[BC] Pre-training complete.")
    print(f"[BC] Load for PPO fine-tuning with: python RL/train.py --env grasp --init-from {save_path}")
    print("=" * 65)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behaviour-clone the expert into a SB3 PPO warm-start."
    )
    parser.add_argument("--rollouts",    type=int,   default=DEFAULT_ROLLOUTS)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--save-path",   type=str,   default=DEFAULT_SAVE_PATH)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    pretrain_bc(
        n_rollouts=args.rollouts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )
