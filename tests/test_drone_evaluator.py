"""
Test for DroneEvaluator: complete episode evaluation with fitness calculation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-pybullet-drones'))

import pytest
import neat
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from src.neat.drone_evaluator import DroneEvaluator
from tests.utils import initialize_population


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
