"""
DroneEvaluator: Evaluates NEAT genomes by running them in HoverAviary environment.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gym-pybullet-drones'))

import numpy as np
import neat
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class DroneEvaluator:
    """
    Evaluates NEAT genomes by controlling a drone in HoverAviary environment.
    """
    MAX_RPM = 21702
    SURVIVAL_BONUS_FACTOR = 100.0

    def __init__(self, gui=False, max_steps=None):
        """
        Initialize the DroneEvaluator.

        Args:
            gui: If True, show PyBullet GUI
            max_steps: Maximum steps per episode (None = use environment default)
        """
        self.gui = gui
        self.max_steps = max_steps

    def _get_env_config(self) -> dict:
        """Helper for environment configuration."""
        return {
            "gui": self.gui,
            "drone_model": DroneModel.CF2X,
            "physics": Physics.PYB,
            "obs": ObservationType.KIN,
            "act": ActionType.RPM
        }
    
    def _map_action(self, output: list[float]) -> np.ndarray:
        """
        Map network output [-1, 1] to RPM [0, MAX_RPM].
        
        Args:
            output: raw network output list.

        Returns:
            np.ndarray: Mapped and clipped action (shape: (1, 4)).
        """
        action = np.array(output)
        # Mapping from [-1, 1] to [0, MAX_RPM]
        action = (action + 1.0) / 2.0 * self.MAX_RPM
        # Clipping for safety (even though mapping should suffice)
        action = np.clip(action, 0, self.MAX_RPM)
        
        return action.reshape(1, 4)

    def _calculate_fitness(self, total_reward: float, steps: int, max_steps: int) -> float:
            """
            Calculate the final fitness value.

            Fitness = total reward + survival bonus (scaled by completion percentage)
            """
            survival_bonus = (steps / max_steps) * self.SURVIVAL_BONUS_FACTOR
            return total_reward + survival_bonus

    def evaluate_genome(self, genome, config):
        """
        Evaluate a single NEAT genome by running it in HoverAviary.

        Args:
            genome: NEAT genome to evaluate
            config: NEAT configuration object

        Returns:
            float: Fitness value (higher is better)
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        with HoverAviary(**self._get_env_config()) as env:

            if self.max_steps is not None:
                max_steps = self.max_steps
            else:
                max_steps = int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)

            obs, info = env.reset()

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated) and steps < max_steps:
                
                state = obs[0] 
                
                output = net.activate(state.tolist())

                action = self._map_action(output)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

            fitness = self._calculate_fitness(total_reward, steps, max_steps)
            
            return fitness