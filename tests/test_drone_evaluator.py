"""
Test for DroneEvaluator: complete episode evaluation with fitness calculation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-pybullet-drones'))

import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from src.neat.drone_evaluator import DroneEvaluator
from tests.utils import initialize_population


def test_get_env_config():
    """Test that _get_env_config returns correct configuration."""
    evaluator = DroneEvaluator(gui=False)
    config = evaluator._get_env_config()

    assert isinstance(config, dict), "Config should be a dictionary"
    assert config["gui"] == False, "GUI should be False"
    assert config["drone_model"] == DroneModel.CF2X, "Drone model should be CF2X"
    assert config["physics"] == Physics.PYB, "Physics should be PYB"
    assert config["obs"] == ObservationType.KIN, "Observation type should be KIN"
    assert config["act"] == ActionType.RPM, "Action type should be RPM"


def test_map_action_zero():
    """Test mapping of zero output (middle of range)."""
    evaluator = DroneEvaluator(gui=False)
    output = [0.0, 0.0, 0.0, 0.0]

    action = evaluator._map_action(output)

    expected_rpm = evaluator.MAX_RPM / 2.0
    assert action.shape == (1, 4), f"Expected shape (1, 4), got {action.shape}"
    np.testing.assert_allclose(action[0], [expected_rpm] * 4, rtol=1e-5)


def test_calculate_fitness_with_reward():
    """Test fitness calculation with reward."""
    evaluator = DroneEvaluator(gui=False)

    fitness = evaluator._calculate_fitness(
        total_reward=42.5,
        steps=150,
        max_steps=200
    )

    # Fitness = 42.5 + (150/200) * 100 = 117.5
    expected = 117.5
    assert fitness == expected, f"Expected fitness {expected}, got {fitness}"


def test_map_action_min_max():
    """Test mapping of minimum and maximum outputs."""
    evaluator = DroneEvaluator(gui=False)

    action_min = evaluator._map_action([-1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(action_min[0], [0.0] * 4, rtol=1e-5)

    action_max = evaluator._map_action([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(action_max[0], [evaluator.MAX_RPM] * 4, rtol=1e-5)


def test_calculate_fitness_no_reward():
    """Test fitness calculation with zero reward."""
    evaluator = DroneEvaluator(gui=False)

    fitness = evaluator._calculate_fitness(
        total_reward=0.0,
        steps=100,
        max_steps=200
    )

    # Fitness = 0 + (100/200) * 100 = 50
    expected = 50.0
    assert fitness == expected, f"Expected fitness {expected}, got {fitness}"


def test_evaluator_initialization():
    """Test that DroneEvaluator can be instantiated."""
    evaluator = DroneEvaluator(gui=False)
    assert evaluator is not None, "Failed to create DroneEvaluator"


def test_evaluate_single_genome():
    """Test that evaluator can evaluate a single genome and return fitness."""
    pop, config = initialize_population()
    genome_id, genome = list(pop.population.items())[0]

    evaluator = DroneEvaluator(gui=False)
    fitness = evaluator.evaluate_genome(genome, config)

    assert isinstance(fitness, (int, float)), f"Expected numeric fitness, got {type(fitness)}"
    assert fitness >= 0, f"Fitness should be non-negative, got {fitness}"


def test_evaluate_multiple_genomes():
    """Test that evaluator can evaluate multiple genomes."""
    pop, config = initialize_population()

    genomes = list(pop.population.items())[:3]  # Test with 3 genomes

    evaluator = DroneEvaluator(gui=False)

    fitnesses = []
    for genome_id, genome in genomes:
        fitness = evaluator.evaluate_genome(genome, config)
        fitnesses.append(fitness)

    assert len(fitnesses) == 3, f"Expected 3 fitness values, got {len(fitnesses)}"
    assert all(isinstance(f, (int, float)) for f in fitnesses), "All fitness values must be numeric"


def test_fitness_deterministic():
    """Test that evaluating the same genome twice gives the same fitness."""
    pop, config = initialize_population()
    _, genome = list(pop.population.items())[0]

    evaluator = DroneEvaluator(gui=False)

    # Evaluate same genome twice
    fitness1 = evaluator.evaluate_genome(genome, config)
    fitness2 = evaluator.evaluate_genome(genome, config)

    # Should get same fitness (deterministic environment)
    assert fitness1 == fitness2, f"Expected deterministic fitness, got {fitness1} and {fitness2}"

