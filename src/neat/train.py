"""NEAT training script for drone control."""

import multiprocessing as mp
import neat
from neat.parallel import ParallelEvaluator
from neat.statistics import StatisticsReporter
from src.neat.drone_evaluator import DroneEvaluator

def eval_genome(genome, config):
    """
    Evaluation function for parallel processing.

    Args:
        genome: NEAT genome to evaluate
        config: NEAT configuration

    Returns:
        Fitness value
    """
    evaluator = DroneEvaluator(gui=False)
    return evaluator.evaluate_single(genome, config)


class NEATTrainer:
    """Handles NEAT training for drone control."""

    def __init__(self, config_path: str, generations: int, checkpoint_interval: int = 10, num_workers: int = None):
        """
        Initialize NEAT trainer.

        Args:
            config_path: Path to NEAT configuration file
            generations: Number of generations to train
            checkpoint_interval: Save checkpoint every N generations
            num_workers: Number of parallel workers (None = use all CPU cores)
        """
        self.config_path = config_path
        self.generations = generations
        self.checkpoint_interval = checkpoint_interval
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()

        # Load NEAT configuration
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        self.population = neat.Population(self.config)

        # Add reporters for statistics and output
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats_reporter = StatisticsReporter()
        self.population.add_reporter(self.stats_reporter)
        self.population.add_reporter(
            neat.Checkpointer(checkpoint_interval, filename_prefix='checkpoints/neat-checkpoint-')
        )

        self.evaluator = DroneEvaluator(gui=False)

        self.parallel_evaluator = ParallelEvaluator(
            self.num_workers,
            eval_genome
        )

    def run(self):
        """
        Run NEAT evolution for specified generations.

        Returns:
            Best genome found during evolution
        """
        winner = self.population.run(self.parallel_evaluator.evaluate, self.generations)
        return winner

    def get_stats(self):
        """
        Get training statistics.

        Returns:
            StatisticsReporter object with training stats
        """
        return self.stats_reporter
