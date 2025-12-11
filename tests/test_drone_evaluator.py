"""
Test for DroneEvaluator: complete episode evaluation with fitness calculation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "gym-pybullet-drones"))

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
    assert config["act"] == ActionType.PID, "Action type should be PID"


def test_map_action_zero():
    """Test mapping of zero output to target position."""
    evaluator = DroneEvaluator(gui=False)
    output = [0.0, 0.0, 0.0]
    current_pos = np.array([0.0, 0.0, 0.5])

    action = evaluator._map_action(output, current_pos)

    # With zero output, target should be [0, 0, 1] (TARGET_POS)
    expected_target = np.array([0.0, 0.0, 1.0])
    assert action.shape == (1, 3), f"Expected shape (1, 3), got {action.shape}"
    np.testing.assert_allclose(action[0], expected_target, rtol=1e-5)


def test_map_action_with_offset():
    """Test mapping of output with offset."""
    evaluator = DroneEvaluator(gui=False)
    output = [1.0, -1.0, 0.5]  # Max x, min y, half z
    current_pos = np.array([0.0, 0.0, 0.5])

    action = evaluator._map_action(output, current_pos)

    # Target = [0, 0, 1] + [0.2, -0.2, 0.1] = [0.2, -0.2, 1.1]
    expected_target = np.array([0.2, -0.2, 1.1])
    assert action.shape == (1, 3), f"Expected shape (1, 3), got {action.shape}"
    np.testing.assert_allclose(action[0], expected_target, rtol=1e-5)


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

