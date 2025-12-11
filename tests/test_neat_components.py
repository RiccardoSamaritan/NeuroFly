"""
Test NEAT genome and network creation and functionality with HoverAviary environment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "gym-pybullet-drones"))

import numpy as np
import neat
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from tests.utils import initialize_population


def test_create_genome_from_config():
    """Test that we can create a NEAT genome from config."""
    pop, config = initialize_population()

    # Get initial population
    genomes = list(pop.population.items())

    assert len(genomes) > 0, "Population should have genomes"
    genome_id, genome = genomes[0]
    assert genome is not None, "Failed to create genome"

def test_create_network_from_genome():
    """Test that we can create a NEAT network from a genome."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]

    # Create network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    assert net is not None, "Failed to create network"


def test_network_input_output_dimensions():
    """Test that network accepts correct input/output dimensions."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Test with 6 inputs (error_pos + drone_vel)
    input_data = np.random.randn(6).tolist()
    output = net.activate(input_data)

    assert len(output) == 3, f"Expected 3 outputs, got {len(output)}"
    assert all(isinstance(o, (int, float)) for o in output), "Output values must be numeric"


def test_network_output_range():
    """Test that network output is in expected range (tanh activation)."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Test multiple random inputs
    for _ in range(10):
        input_data = np.random.randn(6).tolist()
        output = net.activate(input_data)

        for o in output:
            assert -1.5 <= o <= 1.5, f"Output {o} outside expected range [-1.5, 1.5] for tanh"


def test_evaluate_genome_single_step():
    """Test that we can evaluate a genome for one step in HoverAviary with PID."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create environment with PID action
    from gym_pybullet_drones.utils.enums import ActionType
    env = HoverAviary(gui=False, act=ActionType.PID)
    obs, info = env.reset()

    # Extract the 6 inputs needed: error_pos (3) + drone_vel (3)
    TARGET_POS = np.array([0.0, 0.0, 1.0])
    state_raw = obs[0]
    drone_pos = state_raw[0:3]
    drone_vel = state_raw[10:13]
    error_vector = TARGET_POS - drone_pos
    input_data = np.concatenate([error_vector, drone_vel]).tolist()

    output = net.activate(input_data)

    # Map output to target position for PID
    act = np.array(output[:3])
    offset_x = np.clip(act[0], -1.0, 1.0) * 0.2
    offset_y = np.clip(act[1], -1.0, 1.0) * 0.2
    offset_z = np.clip(act[2], -1.0, 1.0) * 0.2
    target = TARGET_POS + np.array([offset_x, offset_y, offset_z])
    target[2] = max(target[2], 0.1)
    action = np.array([target], dtype=np.float32)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, (int, float)), f"Expected numeric reward, got {type(reward)}"
    env.close()
