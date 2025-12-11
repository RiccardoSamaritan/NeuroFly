"""
DroneEvaluator: Evaluates NEAT genomes by running them in HoverAviary environment.
The drone must hover at position [0, 0, 1] using PID control.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gym-pybullet-drones'))

import numpy as np
import neat
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class DroneEvaluator:
    def __init__(self, gui=False, max_steps=None):
        self.gui = gui
        self.max_steps = max_steps

    def _get_env_config(self) -> dict:
        return {
            "gui": self.gui,
            "drone_model": DroneModel.CF2X,
            "physics": Physics.PYB,
            "obs": ObservationType.KIN,
            "act": ActionType.PID
        }

    def _map_action(self, output: list[float], current_pos: np.ndarray) -> np.ndarray:
        """
        Maps network output to target position for PID controller.

        Args:
            output: Network output (3 values: x, y, z offsets)
            current_pos: Current position of the drone

        Returns:
            Target position for the PID controller (shape: (1, 3))
        """
        act = np.array(output[:3])

        # Fixed target position for hovering
        TARGET_POS = np.array([0.0, 0.0, 1.0])

        # Scale output from [-1, 1] to small offsets (max ±0.2m)
        offset_x = np.clip(act[0], -1.0, 1.0) * 0.2
        offset_y = np.clip(act[1], -1.0, 1.0) * 0.2
        offset_z = np.clip(act[2], -1.0, 1.0) * 0.2

        # Final target = target position + offset
        target = TARGET_POS + np.array([offset_x, offset_y, offset_z])

        # Ensure z remains positive
        target[2] = max(target[2], 0.1)

        return np.array([target], dtype=np.float32)

    def evaluate_genome(self, genome, config):
        if isinstance(genome, list):
            for genome_id, g in genome:
                g.fitness = self.evaluate_single(g, config)
            return
        return self.evaluate_single(genome, config)

    def evaluate_single(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)

        TARGET_POS = np.array([0.0, 0.0, 1.0])

        with HoverAviary(**self._get_env_config()) as env:

            if self.max_steps is not None:
                max_steps = self.max_steps
            else:
                max_steps = int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)

            obs, info = env.reset()
            cumulative_fitness = 0.0
            steps = 0

            force_stop = False

            while not force_stop and steps < max_steps:

                # State extraction
                state_raw = obs[0]
                drone_pos = state_raw[0:3]
                drone_vel = state_raw[10:13]

                # Compute error to fixed target
                error_vector = TARGET_POS - drone_pos

                # Input to network (position error + drone velocity)
                inputs = np.concatenate([error_vector, drone_vel])

                # Action
                output = net.activate(inputs)
                action = self._map_action(output, drone_pos)

                # PID controller stabilizes the drone
                obs, _, terminated, truncated, _ = env.step(action)

                # Fitness (Rewards staying close to fixed target)
                dist_error = np.linalg.norm(error_vector)

                # Exponential fitness: 1.0 if error 0, drops quickly
                dist_score = 1.0 / (1.0 + dist_error)

                cumulative_fitness += dist_score

                # Early exit if drone is lost
                if dist_error > 2.5 or drone_pos[2] < 0.05:
                     force_stop = True

                steps += 1

            # Bonus if survives until the end
            if steps >= max_steps:
                cumulative_fitness += 50.0

            return cumulative_fitness