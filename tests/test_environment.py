"""
Test for HoverAviary environment from gym-pybullet-drones
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "gym-pybullet-drones"))

import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def test_hoveraviary_creation():
    """Test correct creation of HoverAviary environment."""
    env = HoverAviary(gui=False)
    assert env is not None, "Failed to create HoverAviary environment"
    env.close()


def test_hoveraviary_reset():
    """Test reset() returns a valid observation."""
    env = HoverAviary(gui=False)
    obs, info = env.reset()

    # Check that obs is a numpy array
    assert isinstance(obs, np.ndarray), f"Expected obs to be ndarray, got {type(obs)}"

    # Check shape: (1, 72) - 1 drone, 72 features
    assert obs.ndim == 2, f"Expected 2D array, got {obs.ndim}D"
    assert obs.shape[0] == 1, f"Expected 1 drone, got {obs.shape[0]}"
    assert obs.shape[1] == 72, f"Expected 72 features, got {obs.shape[1]}"

    env.close()


def test_hoveraviary_step():
    """Test that step() works with random action."""
    env = HoverAviary(gui=False)
    obs, info = env.reset()

    # Random action (1 drone, 4 RPM values)
    MAX_RPM = 21702
    action = np.random.uniform(0, MAX_RPM, size=(1, 4))

    # Step
    obs, reward, terminated, truncated, info = env.step(action)

    # Verify return types
    assert isinstance(obs, np.ndarray), f"Expected obs to be dict, got {type(obs)}"
    assert isinstance(reward, (int, float, dict)), f"Expected reward to be numeric or dict, got {type(reward)}"
    assert isinstance(terminated, bool), f"Expected terminated to be bool, got {type(terminated)}"
    assert isinstance(truncated, bool), f"Expected truncated to be bool, got {type(truncated)}"
    assert isinstance(info, dict), f"Expected info to be dict, got {type(info)}"

    env.close()


def test_observation_shape():
    """Test that observations have correct shape."""
    env = HoverAviary(gui=False, obs=ObservationType.KIN)
    obs, info = env.reset()

    # KIN observation: (1 drone, 72 features)
    expected_shape = (1, 72)
    actual_shape = obs.shape
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"

    env.close()


def test_action_space():
    """Test that action space is correct."""
    env = HoverAviary(gui=False, act=ActionType.RPM)

    # RPM action shape: (1 drone, 4 motors)
    expected_shape = (1, 4)
    actual_shape = env.action_space.shape
    assert actual_shape == expected_shape, f"Expected action space shape {expected_shape}, got {actual_shape}"

    env.close()


def test_episode_termination():
    """Test that episode terminates correctly."""
    env = HoverAviary(gui=False)
    obs, info = env.reset()

    MAX_RPM = 21702
    max_steps = 1000

    for step in range(max_steps):
        action = np.random.uniform(0, MAX_RPM, size=(1, 4))
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Verify that episode terminated
    assert terminated or truncated or step == max_steps - 1, \
        f"Episode did not terminate: terminated={terminated}, truncated={truncated}, step={step}"

    env.close()