"""
Test NEAT genome and network creation and functionality with HoverAviary environment.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-pybullet-drones'))

import numpy as np
import pytest
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

    # Test with 72 inputs (HoverAviary observation size)
    input_data = np.random.randn(72).tolist()
    output = net.activate(input_data)

    assert len(output) == 4, f"Expected 4 outputs, got {len(output)}"
    assert all(isinstance(o, (int, float)) for o in output), "Output values must be numeric"


def test_network_output_range():
    """Test that network output is in expected range (tanh activation)."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Test multiple random inputs
    for _ in range(10):
        input_data = np.random.randn(72).tolist()
        output = net.activate(input_data)

        for o in output:
            assert -1.5 <= o <= 1.5, f"Output {o} outside expected range [-1.5, 1.5] for tanh"


def test_evaluate_genome_single_step():
    """Test that we can evaluate a genome for one step in HoverAviary."""
    pop, config = initialize_population()

    genome_id, genome = list(pop.population.items())[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create environment
    env = HoverAviary(gui=False)
    obs, info = env.reset()

    # Get action from network
    # obs shape: (1, 72), need to flatten to list
    input_data = obs[0].tolist()
    output = net.activate(input_data)

    # Map output from [-1, 1] to RPM [0, MAX_RPM]
    MAX_RPM = 21702
    action = np.array(output)
    action = (action + 1.0) / 2.0 * MAX_RPM
    action = np.clip(action, 0, MAX_RPM)
    action = action.reshape(1, 4)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, (int, float)), f"Expected numeric reward, got {type(reward)}"
    env.close()
