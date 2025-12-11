"""
Main training script for NEAT drone control.

Usage:
    python train_drone.py --generations 50
    python train_drone.py --generations 100 --workers 4
    python train_drone.py --help
"""

import argparse
import pickle
from pathlib import Path
from src.neat.train import NEATTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NEAT algorithm for drone control in HoverAviary environment"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to train (default: 50)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPU cores)"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N generations (default: 10)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config-drone",
        help="Path to NEAT configuration file (default: config-drone)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="best_genome.pkl",
        help="Output file for best genome (default: best_genome.pkl)"
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    print("=" * 60)
    print("NEAT Drone Training")
    print("=" * 60)
    print(f"Configuration file: {args.config}")
    print(f"Generations: {args.generations}")
    print(f"Workers: {args.workers if args.workers else 'All CPU cores'}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    print()

    # Create trainer
    trainer = NEATTrainer(
        config_path=args.config,
        generations=args.generations,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.workers
    )

    print(f"Starting training with {trainer.num_workers} parallel workers...")
    print()

    # Run training
    best_genome = trainer.run()

    # Save best genome
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(best_genome, f)

    print()
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best fitness: {best_genome.fitness:.2f}")
    print(f"Best genome saved to: {args.output}")
    print("=" * 60)

    # Print statistics summary
    stats = trainer.get_stats()
    if stats:
        print()
        print("Training Statistics:")
        print(f"  Final generation mean fitness: {stats.get_fitness_mean()[-1]:.2f}")
        print(f"  Final generation stdev: {stats.get_fitness_stdev()[-1]:.2f}")
        best_ever = max(stats.most_fit_genomes, key=lambda g: g.fitness)
        print(f"  Best fitness overall: {best_ever.fitness:.2f}")


if __name__ == "__main__":
    main()
