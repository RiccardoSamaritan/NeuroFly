"""Tests for NEAT training script."""

import sys
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "gym-pybullet-drones"))

import neat
import multiprocessing as mp
from neat.statistics import StatisticsReporter
from src.neat.train import NEATTrainer
from tests.utils import get_config_path


def test_trainer_initialization():
    """Test that NEATTrainer initializes correctly."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=5)

    assert trainer.config_path == config_path
    assert trainer.generations == 5
    assert trainer.checkpoint_interval > 0
    assert trainer.config is not None
    assert trainer.population is not None


def test_trainer_with_custom_checkpoint_interval():
    """Test trainer with custom checkpoint interval."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=10, checkpoint_interval=2)

    assert trainer.checkpoint_interval == 2


def test_train_single_generation():
    """Test training for a single generation."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=1)

    best_genome = trainer.run()

    assert best_genome is not None
    assert hasattr(best_genome, 'fitness')
    assert best_genome.fitness is not None


def test_train_multiple_generations():
    """Test training for multiple generations."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=3)

    best_genome = trainer.run()

    assert best_genome is not None
    assert best_genome.fitness is not None


def test_stats_reporter_added():
    """Test that statistics reporter is added to population."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=1)

    # Check that StdOutReporter and StatisticsReporter are in reporters
    reporters = trainer.population.reporters.reporters

    has_stdout = any(isinstance(r, neat.reporting.StdOutReporter) for r in reporters)
    has_stats = any(isinstance(r, StatisticsReporter) for r in reporters)

    assert has_stdout, "StdOutReporter should be added"
    assert has_stats, "StatisticsReporter should be added"


def test_get_stats():
    """Test that statistics can be retrieved after training."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=2)

    trainer.run()
    stats = trainer.get_stats()

    assert stats is not None
    assert hasattr(stats, 'get_fitness_mean')
    assert hasattr(stats, 'get_fitness_stdev')


def test_trainer_with_parallel_workers():
    """Test trainer with parallel workers specified."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=1, num_workers=2)

    assert trainer.num_workers == 2


def test_parallel_evaluation():
    """Test that parallel evaluation works correctly."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=1, num_workers=2)

    best_genome = trainer.run()
    
    assert best_genome is not None
    assert best_genome.fitness is not None


def test_trainer_defaults_to_cpu_count():
    """Test that trainer defaults to CPU count when num_workers not specified."""
    config_path = get_config_path()
    trainer = NEATTrainer(config_path, generations=1)

    assert trainer.num_workers == mp.cpu_count()
