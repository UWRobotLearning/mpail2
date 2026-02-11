import os
import time
import torch
import statistics
import gymnasium as gym
import wandb # TODO: does this start wandb?
from collections import deque
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Dict

# from isaaclab.envs import ManagerBasedRLEnv # Ideally runner works for all env types

from .learner import MPAIL2Learner
from .utils import Stats, task_stats, RolloutsVideo

if TYPE_CHECKING:
    from .configs.cfgs import MPAIL2RunnerCfg

class MPAIL2Runner:

    def __init__(self,
        demonstrations: torch.Tensor | Dict[str, torch.Tensor],
        env: gym.Env,
        runner_cfg: 'MPAIL2RunnerCfg',
        device: str = "cuda",
        dtype=torch.float32
    ):
        self.env = env
        self.cfg = runner_cfg
        self.log_cfg = self.cfg.log_cfg
        self.num_envs = env.unwrapped.num_envs
        self.device = device
        self.dtype = dtype
        self.demonstrations = demonstrations

        if type(self.demonstrations) is dict:
            for key in self.demonstrations:
                self.demonstrations[key] = self.demonstrations[key].to(device=self.device, dtype=self.dtype)
                # We assume demonstrations are stored as (N, 2, *obs_shape)
                # where N is flattened trajectories and 2 is (obs, next_obs)
                assert self.demonstrations[key].dim() >= 3 and self.demonstrations[key].shape[1] == 2, \
                    f"Demonstrations for key {key} must have shape (N, 2, *obs_shape), got {self.demonstrations[key].shape}"
        else:
            self.demonstrations = self.demonstrations.to(device=self.device, dtype=self.dtype)
            # We assume demonstrations are stored as (N, 2, *obs_shape)
            # where N is flattened trajectories and 2 is (obs, next_obs)
            assert self.demonstrations.dim() >= 3 and self.demonstrations.shape[1] == 2, \
                f"Demonstrations must have shape (N, 2, *obs_shape), got {self.demonstrations.shape}"

        self.log_dir = self.log_cfg.log_dir
        self._num_steps_per_env = self.env.unwrapped.max_episode_length

        # Create MPAIL Learner
        self.learner = MPAIL2Learner(self.demonstrations, self.num_envs,
                                    self.cfg.learner_cfg, device=device)

        self.learner_cfg = self.cfg.learner_cfg

        # Extract observation shape from environment
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Dict observation space - get shape for each observation group
            actor_obs_shape = {}
            for k in env.observation_space.keys():
                if isinstance(env.observation_space[k], gym.spaces.Dict):
                    # Skip nested dicts (not supported)
                    continue
                actor_obs_shape[k] = env.observation_space[k].shape[1:]  # Remove batch dim
        else:
            # Tensor observation space
            actor_obs_shape = env.observation_space.shape[1:]  # Remove batch dim

        # Use learner's internal action dimension for storage (not env's)
        # This ensures storage uses the actual control dimension, not padded dimension
        self._learner_action_dim = self.learner.planner.sampling.nu

        self.learner.init_storage(
            num_envs=self.num_envs,
            num_steps_per_env=self._num_steps_per_env, # -1 since we store (s_t, a_t, s_t+1)
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=None, # No privileged observations
            action_shape=[self._learner_action_dim],
        )

        # Log
        if self.cfg.logger == "wandb":
            # from rsl_rl.utils.wandb_utils import WandbSummarylogger
            # self.logger = WandbSummarylogger(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            import wandb
            self.logger = wandb
        else:
            print("[INFO] No logger specified or not recognized.")
            self.logger = None

        if self.cfg.logger and self.cfg.vis_rollouts:
            assert self.learner.planner.vis is not None, "Policy must have visualization enabled for rollouts"
            self.rollouts_vid = RolloutsVideo(self.learner.planner.vis)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def train_mode(self):
        self.learner.planner.train()

    def learn(self):

        # start learning
        obs, infos = self.env.reset()
        critic_obs = obs
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.cfg.num_learning_iterations
        for it in tqdm(range(start_iter, tot_iter)):
            start = time.time()
            ep_stats = Stats()

            # Set current iteration on environment for curriculum learning
            self.env.unwrapped.current_iteration = it

            vis_ep = self.cfg.vis_rollouts and it % self.video_interval_its == 0
            # Rollout
            self.learner.eval()
            for i in range(self._num_steps_per_env):

                with torch.inference_mode():

                    # Sample actions from policy
                    actions = self.learner.act(obs, vis_rollouts=vis_ep) # TODO: critic obs

                    # Step environment
                    next_obs, rewards, terms, truncs, infos = self.env.step(actions.to(self.env.unwrapped.device))

                    # Move to the agent device
                    dones = (terms | truncs).to(dtype=torch.long)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    # Process env step and store in buffer
                    stats = self.learner.process_env_step(rewards, dones, infos, next_obs)

                    if self.log_dir is not None:

                        # Log information
                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        # Update step stats
                        ep_stats.update(stats)
                        ep_stats.update(infos['log'])

                        # Clear data for completed episodes
                        # Use 1D indices and guard empty to avoid CUDA device-side asserts
                        new_ids = (dones > 0).nonzero(as_tuple=True)[0]
                        if new_ids.numel() > 0:
                            # cur_reward_sum and cur_episode_length are 1D (num_envs,)
                            rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                if vis_ep:
                    self.rollouts_vid.update_video()

                # Update obs
                obs = next_obs

            # End of rollout collection
            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop

            # Log returns
            ep_ret = self.learner.storage.rewards.sum(0).cpu().numpy()
            max_return, mean_return, min_return = ep_ret.max(), ep_ret.mean(), ep_ret.min()

            # Update learner
            # Note: we keep arguments here since locals() loads them
            train_stats = {}
            self.learner.train()
            train_stats = self.learner.update(iteration=it) # Can return metrics for loss logging

            self.learner.storage.clear()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # TODO
            # Logging info and save checkpoint

            fps = int(self._num_steps_per_env * self.num_envs / (collection_time + learn_time))
            stats = {
                "Perf/collection_time": collection_time,
                "Perf/learn_time": learn_time,
                "Perf/fps": fps,
                "Env/mean_reward": 0 if len(rewbuffer) == 0 else statistics.mean(rewbuffer),
                "Env/mean_length": 0 if len(lenbuffer) == 0 else statistics.mean(lenbuffer),
                "Env/max_return": max_return,
                "Env/mean_return": mean_return,
                "Env/min_return": min_return,
                "it": it,
                "tot_iter": tot_iter,
            }

            train_stats.update(stats)
            train_stats.update(ep_stats.mean())

            if self.log_dir is not None and self.logger:

                self.logger.log(train_stats)

                if hasattr(self.cfg, "log_task_stat"):
                    # Log task-specific statistics
                    task_stat = task_stats(ep_stats.stats, self.cfg.log_task_stat)
                    train_stats.update(task_stat)

                # Log Rollout visualization
                if vis_ep:
                    vid_save_dir = os.path.join(self.log_dir, "rollouts_vis")
                    path_to_vid = self.rollouts_vid.save_video(
                        output_dir=vid_save_dir,
                        episode_num=it,
                        frame_rate=10
                    )
                    if not self.log_cfg.no_wandb:
                        wandb.log({"Rollouts Video": wandb.Video(path_to_vid)}, commit=False)
                        self.rollouts_vid.reset()

                # Save model
                if it % self.log_cfg.checkpoint_every == 0:
                    self.save(postfix=f"{it}")

        # Save the final model after training
        if self.log_dir is not None:
            self.save(postfix=f"{tot_iter}")

    def save(self, postfix: str=""):

        saved_dict = {
            "model_state_dict": self.learner.planner.state_dict(),
            "reward_optimizer_state_dict": self.learner._reward_opt.state_dict(),
            "value_optimizer_state_dict": self.learner._value_opt.state_dict(),
            "iter": self.current_learning_iteration,
        }

        model_path = os.path.join(self.log_dir, "models", f"model_{postfix}.pt")
        torch.save(saved_dict, model_path)

        # Save to external logger
        if self.logger:
            self.logger.save(model_path, base_path = os.path.dirname(model_path))


    def load(self, path: str, load_optimizer=True):
        ''' Load model from path '''

        saved_dict = torch.load(path, map_location=self.device)

        self.learner.planner.load_state_dict(saved_dict["model_state_dict"])
        if load_optimizer: # Does this point to the correct params?
            self.learner._reward_opt.load_state_dict(saved_dict["reward_optimizer_state_dict"])
            self.learner._value_opt.load_state_dict(saved_dict["value_optimizer_state_dict"])

    @property
    def video_interval_its(self):
        return self.log_cfg.video_interval // self._num_steps_per_env