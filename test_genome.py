#!/usr/bin/env python3
"""
Drone Viewer: Visualizza il comportamento di un genoma NEAT addestrato.
Nessun calcolo di fitness, solo visualizzazione pura.
"""

import sys
import os
import time
import argparse
import pickle
import numpy as np
import pybullet as p
import neat
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "gym-pybullet-drones"))

from src.neat.drone_evaluator import DroneEvaluator
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

class DroneViewer(DroneEvaluator):
    """
    Class to visualize the behavior of a NEAT-trained drone genome.
    Inherits from DroneEvaluator to reuse environment setup and action mapping.
    """
    
    def show_behavior(self, genome, config):
        try:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        except (AttributeError, TypeError):
            net = neat.nn.FeedForwardNetwork.create(genome, config)

        print(f"--- START OF VISUAL DEMO ---")

        env_config = self._get_env_config()
        env_config["gui"] = True
        
        env = HoverAviary(**env_config)
        
        try:
            while True: 
                obs, info = env.reset()
                steps = 0

                target_pos = np.array([0.0, 0.0, 1.0])
                
                print("\nNew Run...")
                print("Press Ctrl+C to exit.")

                max_steps = int(30 * env.CTRL_FREQ)

                while steps < max_steps:
                    state_raw = obs[0]
                    drone_pos = state_raw[0:3]
                    drone_vel = state_raw[10:13]

                    error_vector = target_pos - drone_pos
                    inputs = np.concatenate([error_vector, drone_vel])

                    output = net.activate(inputs)
                    action = self._map_action(output, drone_pos)

                    obs, _, terminated, truncated, _ = env.step(action)

                    p.addUserDebugLine(
                        lineFromXYZ=target_pos,
                        lineToXYZ=target_pos + np.array([0, 0, 0.2]),
                        lineColorRGB=[1, 0, 0],
                        lifeTime=0.1,
                        physicsClientId=env.CLIENT
                    )

                    p.addUserDebugLine(
                        lineFromXYZ=drone_pos,
                        lineToXYZ=drone_pos, 
                        lineColorRGB=[0, 1, 0],
                        lifeTime=1.5,
                        lineWidth=2,
                        physicsClientId=env.CLIENT
                    )

                    time.sleep(1/30)

                    if steps % 10 == 0:
                        dist = np.linalg.norm(error_vector)
                        print(f"\rDist: {dist:.2f}m | Z: {drone_pos[2]:.2f}m", end="")

                   
                    dist = np.linalg.norm(error_vector)
                    if drone_pos[2] < 0.05 or dist > 2.0:
                        print(f"\nCRASH!. Resetting...")
                        break 

                    steps += 1
                
                print("\n--- Run Ended ---\n")

        except KeyboardInterrupt:
            print("\n\n--- DEMO TERMINATED BY USER ---\n")
        finally:
            env.close()

def main():
    parser = argparse.ArgumentParser(description="Show NEAT genome behavior")
    parser.add_argument("genome_file", type=str, help="Path to .pkl file")
    parser.add_argument("--config", type=str, default="config-drone", help="Path to config file")
    args = parser.parse_args()

    # Load Config
    if not os.path.exists(args.config):
        print("Config file not found.")
        return
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)

    # Load Genome
    if not os.path.exists(args.genome_file):
        print("Genome file not found.")
        return
    with open(args.genome_file, 'rb') as f:
        genome = pickle.load(f)

    # Run Viewer
    viewer = DroneViewer()
    viewer.show_behavior(genome, config)

if __name__ == "__main__":
    main()