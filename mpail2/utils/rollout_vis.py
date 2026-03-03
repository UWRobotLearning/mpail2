import torch
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from io import BytesIO
import imageio
from typing import TYPE_CHECKING

import math

from numpy.ma import masked_array

from scipy.ndimage import rotate

if TYPE_CHECKING:
    from mpail2.configs.cfgs import RolloutVisConfig

class RolloutsVisualization:
    """
    Visualizes a hierarchical animation where new rollouts are added at each simulation timestep.

    Parameters:
        sim_timesteps (int): Total number of simulation timesteps.
        data_generator (callable): Function that provides rollout data at each simulation timestep.
                                   Expected output: [num_envs, num_rollouts, num_timestep, state_dim].
        env_ids (list): List of environment indices to visualize.
        rollout_ids (list): List of rollout indices to visualize.
        performance_metrics_fn (callable): Function to compute performance metrics.
                                           Input: [num_envs, num_rollouts].
                                           Output: [num_envs, num_rollouts].
        save_path (str): Path to save the generated video.
        trail_length (int): Length of the trajectory trail to display.
    """
    def __init__(self,
        cfg: 'RolloutVisConfig',
        map_res_m_px: float=0.1,
        map_length_px: int=256,
    ):
        self.cfg = cfg
        self.vis_n_envs = cfg.vis_n_envs
        self.vis_n_rollouts = cfg.vis_n_rollouts
        self.xlim = cfg.xlim
        self.ylim = cfg.ylim
        self.map_res_m_px = map_res_m_px
        self.map_length_px = map_length_px
        self.show_velocity = cfg.show_velocity
        self.show_elevation = cfg.show_elevation
        self.cost_range = cfg.cost_range
        self.show_trajectory_trace = cfg.show_trajectory_trace
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Store trajectory traces for each environment

        # State index configuration
        self.pos_x_idx = cfg.pos_x_idx
        self.pos_y_idx = cfg.pos_y_idx
        self.pos_z_idx = cfg.pos_z_idx
        self.vel_x_idx = cfg.vel_x_idx
        self.vel_y_idx = cfg.vel_y_idx
        self.yaw_idx = cfg.yaw_idx
        self.control_scatter_1_indices = cfg.control_scatter_1_indices
        self.control_scatter_2_indices = cfg.control_scatter_2_indices
        self.control_scatter_3_indices = cfg.control_scatter_3_indices
        self.rollout_x_idx = cfg.rollout_x_idx
        self.rollout_y_idx = cfg.rollout_y_idx
        self.rollout_z_idx = cfg.rollout_z_idx
        self.cube_x_idx = cfg.cube_x_idx
        self.cube_y_idx = cfg.cube_y_idx
        self.cube_z_idx = cfg.cube_z_idx
        self.cube_pos_x_idx = cfg.cube_pos_x_idx
        self.cube_pos_y_idx = cfg.cube_pos_y_idx

        self.reset()

    def update(
        self,
        x0: np.ndarray,
        rollouts: np.ndarray,
        rollout_costs: np.ndarray = None,
        rollout_rewards: np.ndarray = None,
        elevation_map: np.ndarray = None,
        optimal_control: np.ndarray = None,
        frame_timestep: int=None
    ):
        '''
        x0: [num_envs, state_dim]
        rollouts: [num_envs, num_rollouts, num_timestep, state_dim]
        rollout_costs: [num_envs, num_rollouts, num_timesteps]
        rollout_rewards: [num_envs, num_rollouts, num_timesteps] (alternative to costs)
        metrics: [num_envs, num_rollouts]
        vis_rollout_ids: list of rollout indices to visualize
        frame_timestep: current simulation timestep
        '''
        # Support both rollout_costs and rollout_rewards
        rollout_values = rollout_rewards if rollout_rewards is not None else rollout_costs

        # Update colorbar limits only if cost_range is not provided
        if self.cost_range is None:
            self.sm.set_clim(rollout_values.min(), rollout_values.max())
        else:
            self.sm.set_clim(*self.cost_range)

        # normalize elevation map
        if self.show_elevation and elevation_map is not None:
            elevation_map = elevation_map[..., 0]
            elevation_map = (elevation_map - elevation_map.min()) / (elevation_map.max() - elevation_map.min() + 1e-5)

        # Update visuals for each environment
        for i in range(self.vis_n_envs):
            env_rollouts = rollouts[i]  # [num_rollouts, num_timestep, state_dim]

            # Update scatter for the last point using configured indices
            curr_x = x0[i, self.pos_x_idx]
            curr_y = x0[i, self.pos_y_idx]
            curr_z = x0[i, self.pos_z_idx]

            vx = env_rollouts[0, 0, self.vel_x_idx]
            vy = env_rollouts[0, 0, self.vel_y_idx]

            # plot optimal control rollout (using y-z dimensions)
            if optimal_control is not None:
                optimal_control_rollout = optimal_control[i]  # [num_timesteps, action_dim]
                horizon_len = optimal_control_rollout.shape[0]

                # Create alpha values that decay from 1.0 to 0.2 across the horizon
                alphas = np.linspace(1.0, 0.2, horizon_len)

                # Create colors with varying alpha for first scatter (green)
                colors_1 = np.tile([0, 0.5, 0, 1], (horizon_len, 1))  # green in RGBA
                colors_1[:, 3] = alphas

                self.env_control_scatters[i].set_offsets(optimal_control_rollout[:, list(self.control_scatter_1_indices)])
                self.env_control_scatters[i].set_color(colors_1)

                # plot optimal cube control rollout on left plot using configured indices
                colors_cube_left = np.tile([1, 0.647, 0, 1], (horizon_len, 1))  # orange in RGBA
                colors_cube_left[:, 3] = alphas * 0.6  # Base alpha of 0.6

                self.env_control_scatters_cube_left[i].set_offsets(optimal_control_rollout[:, list(self.cfg.control_scatter_cube_left_indices)])
                self.env_control_scatters_cube_left[i].set_color(colors_cube_left)

                # plot second optimal control rollout using configured indices
                colors_2 = np.tile([1, 0.647, 0, 1], (horizon_len, 1))  # orange in RGBA
                colors_2[:, 3] = alphas * 0.6  # Base alpha of 0.6

                self.env_control_scatters_2[i].set_offsets(optimal_control_rollout[:, list(self.control_scatter_2_indices)])
                self.env_control_scatters_2[i].set_color(colors_2)

                # plot third optimal control rollout using configured indices
                colors_3 = np.tile([0, 0.5, 0, 1], (horizon_len, 1))  # green in RGBA
                colors_3[:, 3] = alphas * 0.6  # Base alpha of 0.6

                self.env_control_scatters_3[i].set_offsets(optimal_control_rollout[:, list(self.control_scatter_3_indices)])
                self.env_control_scatters_3[i].set_color(colors_3)

            #plot velocity
            if self.show_velocity:
                velocity = np.sqrt(vx**2 + vy**2)
                self.velocities[i] = np.append(self.velocities[i], velocity)
                self.velocities_lines[i].set_data(range(len(self.velocities[i])), self.velocities[i])
                self.axes[2].relim()
                self.axes[2].autoscale_view()

            yaw = env_rollouts[0, 0, self.yaw_idx]

            yaw = np.nan_to_num(yaw, nan=0)
            rotate_angle = -yaw * 180 / math.pi

            self.env_pos_scatters[i].set_offsets(np.c_[curr_x, curr_z])

            if self.show_elevation and elevation_map is not None:
                env_elevation = elevation_map[i] # [256, 256], only visualizes 1 car
                env_elevation += 1 # shift the elevation map to be positive

                env_elevation = np.nan_to_num(env_elevation, nan=0)
                env_elevation = rotate(env_elevation, angle=rotate_angle, order=3, reshape=True, mode="constant")
                self.elevation_images[i].set_data(env_elevation)
                new_min = self.min_elevation_map_x * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
                new_max = new_min * -1

                self.elevation_images[i].set_extent([new_min + curr_x, new_max + curr_x, new_min + curr_y, new_max + curr_y])

            # Update trajectory lines for LEFT plot (axes[0])
            env_scatters: list = self.env_rollout_scatters[i]
            # Update trajectory scatter using configured indices
            all_x_vals = np.concatenate([env_rollouts[j][:, self.rollout_x_idx] for j in range(self.vis_n_rollouts)])
            all_y_vals = np.concatenate([env_rollouts[j][:, self.rollout_y_idx] for j in range(self.vis_n_rollouts)])
            all_z_vals = np.concatenate([env_rollouts[j][:, self.rollout_z_idx] for j in range(self.vis_n_rollouts)])

            # Update colors
            env_rollout_values = rollout_values[i]  # [num_rollouts]
            all_colors = np.concatenate([self.sm.to_rgba(env_rollout_values[j]) for j in range(self.vis_n_rollouts)])

            env_scatters.set_offsets(np.c_[all_x_vals, all_z_vals])
            env_scatters.set_color(all_colors)

            # Use configured indices for cube/object position
            all_x_vals_cube = np.concatenate([env_rollouts[j][:, self.cube_x_idx] for j in range(self.vis_n_rollouts)])
            all_y_vals_cube = np.concatenate([env_rollouts[j][:, self.cube_y_idx] for j in range(self.vis_n_rollouts)])
            all_z_vals_cube = np.concatenate([env_rollouts[j][:, self.cube_z_idx] for j in range(self.vis_n_rollouts)])

            # Update cube rollouts on LEFT plot (axes[0])
            env_scatters_cube_left = self.env_rollout_scatters_cube_left[i]
            env_scatters_cube_left.set_offsets(np.c_[all_x_vals_cube, all_z_vals_cube])
            env_scatters_cube_left.set_color(all_colors)

            # Update trajectory lines for RIGHT plot (axes[1]) - Cube position
            env_scatters_cube = self.env_rollout_scatters_cube[i]
            env_scatters_cube.set_offsets(np.c_[all_x_vals_cube, all_y_vals_cube])
            env_scatters_cube.set_color(all_colors)  # Use same colors as left plot

            # Update current cube position marker on RIGHT plot
            curr_cube_x = x0[i, self.cube_pos_x_idx]
            curr_cube_y = x0[i, self.cube_pos_y_idx]
            self.env_pos_scatters_right[i].set_offsets(np.c_[curr_cube_x, curr_cube_y])

            # Add current position to trajectory trace using configured indices
            if self.show_trajectory_trace:
                self.trajectory_traces[i].append((curr_x, curr_z))  # Using configured pos_x_idx, pos_z_idx for left plot
                trace_x, trace_z = zip(*self.trajectory_traces[i])
                if len(self.env_rollout_lines[i]) == 0:
                    # Create a new line if it doesn't exist
                    line, = self.axes[0].plot(trace_x, trace_z, 'b-', alpha=0.5, lw=1)
                    self.env_rollout_lines[i].append(line)
                else:
                    # Update the existing line
                    self.env_rollout_lines[i][0].set_data(trace_x, trace_z)

                # Add trajectory trace on RIGHT plot for cube position
                self.trajectory_traces_right[i].append((curr_cube_x, curr_cube_y))
                trace_cube_x, trace_cube_y = zip(*self.trajectory_traces_right[i])
                if len(self.env_rollout_lines_right[i]) == 0:
                    # Create a new line if it doesn't exist
                    line_right, = self.axes[1].plot(trace_cube_x, trace_cube_y, 'b-', alpha=0.5, lw=1)
                    self.env_rollout_lines_right[i].append(line_right)
                else:
                    # Update the existing line
                    self.env_rollout_lines_right[i][0].set_data(trace_cube_x, trace_cube_y)

        # Update the time text
        if frame_timestep is not None:
            self.timestep = frame_timestep
            self.time_text.set_text(f"Simulation Timestep: {self.timestep}")

        # Update boundaries
        if self.xlim is None or self.ylim is None:
            current_xlim = self.axes[0].get_xlim()
            current_ylim = self.axes[0].get_ylim()
            new_xlim = (min(current_xlim[0], rollouts[..., 0].min() - 1), max(current_xlim[1], rollouts[..., 0].max() + 1))
            new_ylim = (min(current_ylim[0], rollouts[..., 1].min() - 1), max(current_ylim[1], rollouts[..., 1].max() + 1))
            self.axes[0].set_xlim(new_xlim)
            self.axes[0].set_ylim(new_ylim)

    def save_frame(self, output_dir, step: int=None):
        """
        Saves the current frame to disk.
        """
        fname = f"rollouts_vis.png"
        if step is not None:
            fname = f"step-{step}_{fname}"

        path = os.path.join(output_dir, fname)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fig.savefig(path)

    def reset(self):
        ''' Set up plot features '''

        if self.show_velocity:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(24, 8))
        else:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))

        # Add colorbar to visualize the normalized performance metrics
        self.cmap = get_cmap("seismic_r")
        if self.cost_range is None:
            norm = mpl.colors.Normalize(vmin=0, vmax=30)  # Default normalization
        else:
            norm = mpl.colors.Normalize(vmin=self.cost_range[0], vmax=self.cost_range[1])  # Use cost_range
        self.sm = mpl.cm.ScalarMappable(
            cmap=self.cmap,
            norm=norm
        )
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=self.axes[0], orientation='vertical',
                          label="Reward")
        self.time_text = self.axes[0].text(0.05, 0.95, '', transform=self.axes[0].transAxes,
                                      fontsize=12, verticalalignment='top')

        self.min_elevation_map_x = -self.map_length_px * self.map_res_m_px/2
        self.max_elevation_map_x = -1*self.min_elevation_map_x
        # no need for y min and max since the height scan is a square
        if self.show_elevation:
            self.elevation_images = []
            for i in range(self.vis_n_envs):
                self.elevation_images.append(self.axes[0].imshow(np.zeros((256,256)), cmap='Greys',
                                                                 extent=(self.min_elevation_map_x, self.max_elevation_map_x,
                                                                         self.min_elevation_map_x, self.max_elevation_map_x),
                                                                         alpha=0.5, origin='lower', vmin=0, vmax=1))

        if self.xlim:
            self.axes[0].set_xlim(self.xlim)
        if self.ylim:
            self.axes[0].set_ylim(self.ylim)

        self.axes[0].set_title("Rollout Visualization")
        self.axes[0].set_xlabel("EE_X")
        self.axes[0].set_ylabel("EE_Z")
        self.axes[0].set_aspect('equal')
        self.axes[0].grid(True)  # Add grid lines

        # Setup second plot (copy of first plot for control dims 6:8)
        self.axes[1].set_xlim([0.2, 0.8])
        self.axes[1].set_ylim([-0.4, 0.4])

        self.axes[1].set_title("Cube Position Visualization")
        self.axes[1].set_xlabel("Cube X")
        self.axes[1].set_ylabel("Cube Y")
        self.axes[1].set_aspect('equal')
        self.axes[1].grid(True)  # Add grid lines

        self.timestep = 0
        self.env_pos_scatters = {} # list of scatter:pos for each env
        self.env_footprint_plots = {} # list of plots of the footprint of each env
        self.env_control_scatters = {} # list of optimal control rollouts for each env (dims 1:3)
        self.env_control_scatters_cube_left = {} # list of optimal cube control rollouts for each env on left plot
        self.env_control_scatters_2 = {} # list of optimal control rollouts for each env (dims 6:8)
        self.env_control_scatters_3 = {} # list of optimal control rollouts for each env (dims 0:2)
        self.env_rollout_lines = {} # list of lines for each env
        self.env_prev_state_hist = []
        self.env_rollout_scatters = {}  # list of scatters for each env (LEFT plot)
        self.env_rollout_scatters_cube = {}  # list of scatters for cube (RIGHT plot)
        self.env_rollout_scatters_cube_left = {}  # list of scatters for cube (LEFT plot)
        self.env_pos_scatters_right = {}  # list of scatter:pos for cube on right plot
        self.env_rollout_lines_right = {}  # list of lines for trajectory on right plot
        self.trajectory_traces = [[] for _ in range(self.vis_n_envs)]  # Reset trajectory traces
        self.trajectory_traces_right = [[] for _ in range(self.vis_n_envs)]  # Trajectory traces for right plot

        # Line plot for velocity (norm(x^2 + y^2))
        if self.show_velocity:
            self.axes[2].set_title("Velocity")
            self.axes[2].set_xlabel("Timestep")
            self.axes[2].set_ylabel("Velocity")
            self.velocities = []
            self.velocities_lines = []

        self.axes[0].legend()

        for i in range(self.vis_n_envs):
            scatter = self.axes[0].scatter([], [], s=100)  # Scatter for current states (increased size)
            self.env_pos_scatters[i] = scatter
            self.env_rollout_lines[i] = []
            self.env_rollout_scatters[i] = self.axes[0].scatter([], [], alpha=0.5, s=20)  # Scatter for trajectory (increased size)

            # Add rollouts scatter for cube on LEFT plot
            self.env_rollout_scatters_cube_left[i] = self.axes[0].scatter([], [], alpha=0.5, s=20, marker='s')  # Scatter for cube trajectory on left plot (square markers)

            # Add rollouts scatter for RIGHT plot (cube position)
            self.env_rollout_scatters_cube[i] = self.axes[1].scatter([], [], alpha=0.5, s=20)  # Scatter for cube trajectory

            # Add current position scatter for RIGHT plot (cube position)
            scatter_right = self.axes[1].scatter([], [], s=100, marker='s')  # Scatter for current cube position
            self.env_pos_scatters_right[i] = scatter_right
            self.env_rollout_lines_right[i] = []  # Lines for trajectory trace on right plot

            if self.show_velocity:
                self.velocities.append(np.array([]))
                line = self.axes[2].plot([], [], label=f'Car {i}')[0]  # Line for velocity
                self.velocities_lines.append(line)

            optimal_control_scatter = self.axes[0].scatter(
                [], [], c='green', alpha=1., s=20, edgecolors='black', lw=0.4
            )  # Scatter for optimal control (dims 1:3)
            self.env_control_scatters[i] = optimal_control_scatter

            optimal_control_scatter_cube_left = self.axes[0].scatter(
                [], [], c='orange', alpha=0.6, s=20, edgecolors='black', lw=0.4, marker='s'
            )  # Scatter for optimal cube control on left plot
            self.env_control_scatters_cube_left[i] = optimal_control_scatter_cube_left

            optimal_control_scatter_2 = self.axes[1].scatter(
                [], [], c='orange', alpha=0.6, s=20, edgecolors='black', lw=0.4
            )  # Scatter for optimal control (dims 6:8)
            self.env_control_scatters_2[i] = optimal_control_scatter_2

            optimal_control_scatter_3 = self.axes[1].scatter(
                [], [], c='green', alpha=0.6, s=20, edgecolors='black', lw=0.4
            )  # Scatter for optimal control (dims 0:2)
            self.env_control_scatters_3[i] = optimal_control_scatter_3

        # Adjust the size of the velocity plot
        if self.show_velocity:
            box = self.axes[2].get_position()
            self.axes[2].set_position([box.x0, box.y0, box.width * 0.5, box.height * 0.5])

    def close(self):
        """
        Closes the visualization.
        """
        plt.close(self.fig)

class RolloutsVideo:
    def __init__(self, rv: RolloutsVisualization):
        self.rv = rv
        self.reset()

    def reset(self):
        self.rv.reset()
        self.img_frames = []
        self.num_resets = 0 # HACK: internal tracker

    def update(self, rollouts: np.ndarray, rollout_costs: np.ndarray,
               frame_timestep: int=None):
        self.rv.update(rollouts, rollout_costs, frame_timestep)
        self.update_video()
        self.rv.close()

    def update_video(self):
        """
        Updates the video with the latest frame.
        """
        # Redraw canvas for updates
        buf = BytesIO()
        self.rv.fig.savefig(buf, format='png')
        buf.seek(0)
        self.img_frames.append(imageio.imread(buf))
        buf.close()

    def save_video(self, output_dir, episode_num: int=None, frame_rate: int=10):
        """
        Combines frames stored in memory into a video and saves it to disk.
        """
        fname = f"rollouts_video.mp4"
        if episode_num is not None:
            fname = f"episode-{episode_num}_{fname}"

        path = os.path.join(output_dir, fname)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.img_frames, fps=frame_rate)
        print(f"Video saved to {path}")
