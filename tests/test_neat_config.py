"""
Test for NEAT configuration.
"""
import os
import pytest
import neat
from tests.utils import get_config_path

def test_config_file_exists():
    """Test that NEAT config file exists."""
    config_path = get_config_path()
    assert os.path.exists(config_path), f"Config file not found at {config_path}"


def test_config_loads():
    """Test that NEAT config loads without errors."""
    config_path = get_config_path()

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        assert config is not None, "Config object is None"
    except Exception as e:
        pytest.fail(f"Failed to load NEAT config: {e}")

def test_config_input_output():
    """Test that config has correct input/output dimensions."""
    config_path = get_config_path()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Check input/output sizes
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs

    assert num_inputs == 6, f"Expected 6 inputs (error_pos + drone_vel), got {num_inputs}"
    assert num_outputs == 3, f"Expected 3 outputs (target x,y,z for PID), got {num_outputs}"


def test_config_population_size():
    """Test that population size is valid."""
    config_path = get_config_path()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop_size = config.pop_size
    assert pop_size > 0, f"Expected pop_size > 0, got {pop_size}"
    assert pop_size >= 10, f"Population too small: {pop_size}, recommend >= 10"

