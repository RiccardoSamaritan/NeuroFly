import os
import neat

def get_config_path():
    """Get path to NEAT config file."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    return os.path.join(project_root, "config-drone")

def initialize_population():
    """Utility function to initialize a NEAT population from config file."""
    config_path = get_config_path()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    return pop, config