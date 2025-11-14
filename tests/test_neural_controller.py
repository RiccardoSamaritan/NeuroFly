"""
Test suite for NeuralController
"""
from pathlib import Path
import pytest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import NeuralController


class TestNeuralController:
    """Test suite for NeuralController class"""

    def test_initialization(self):
        """Test controller initialization"""
        controller = NeuralController(
            input_size=20,
            hidden_sizes=[32, 16],
            output_size=4
        )

        assert controller.input_size == 20
        assert controller.hidden_sizes == [32, 16]
        assert controller.output_size == 4
        assert controller.num_params > 0

    def test_default_parameters(self):
        """Test default parameters match CtrlAviary"""
        controller = NeuralController()

        assert controller.input_size == 20  # CtrlAviary observation
        assert controller.output_size == 4  # 4 motors
        assert isinstance(controller.hidden_sizes, list)

    def test_forward_pass(self):
        """Test forward pass with random input"""
        controller = NeuralController()
        obs = torch.randn(20)

        output = controller.forward(obs)

        assert output.shape == (4,)
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    def test_get_action(self):
        """Test get_action method"""
        controller = NeuralController()

        # Test with numpy array
        obs = np.random.randn(20)
        action = controller.get_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1, 4)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_get_action_with_batch(self):
        """Test get_action with batch dimension"""
        controller = NeuralController()

        obs = np.random.randn(1, 20)
        action = controller.get_action(obs)

        assert action.shape == (1, 4)

    def test_get_set_params(self):
        """Test getting and setting parameters"""
        controller = NeuralController()

        # Get params
        params = controller.get_params()

        assert isinstance(params, np.ndarray)
        assert params.ndim == 1
        assert len(params) == controller.num_params

        # Set new params
        new_params = np.random.randn(controller.num_params) * 0.1
        controller.set_params(new_params)

        # Verify params were set
        retrieved_params = controller.get_params()
        assert np.allclose(retrieved_params, new_params)

    def test_set_params_wrong_size(self):
        """Test that setting wrong size params raises error"""
        controller = NeuralController()

        with pytest.raises(ValueError):
            wrong_params = np.random.randn(100)  # Wrong size
            controller.set_params(wrong_params)

    def test_clone(self):
        """Test cloning controller"""
        controller = NeuralController()

        # Set random params
        params = np.random.randn(controller.num_params) * 0.5
        controller.set_params(params)

        # Clone
        controller2 = controller.clone()

        # Test same architecture
        assert controller2.input_size == controller.input_size
        assert controller2.hidden_sizes == controller.hidden_sizes
        assert controller2.output_size == controller.output_size

        # Test same parameters
        assert np.allclose(controller2.get_params(), controller.get_params())

        # Test same output for same input
        obs = np.random.randn(20)
        action1 = controller.get_action(obs)
        action2 = controller2.get_action(obs)
        assert np.allclose(action1, action2)

    def test_save_load(self, tmp_path: Path):
        """Test saving and loading controller"""
        controller = NeuralController()

        # Set random params
        params = np.random.randn(controller.num_params) * 0.3
        controller.set_params(params)

        # Save
        filepath = tmp_path / "controller.pth"
        controller.save(filepath)

        # Load
        loaded_controller = NeuralController.load(filepath)

        # Test same architecture
        assert loaded_controller.input_size == controller.input_size
        assert loaded_controller.hidden_sizes == controller.hidden_sizes
        assert loaded_controller.output_size == controller.output_size

        # Test same parameters
        assert np.allclose(loaded_controller.get_params(), controller.get_params())

    def test_deterministic_output(self):
        """Test that same input gives same output (no randomness)"""
        controller = NeuralController()

        obs = np.random.randn(20)

        action1 = controller.get_action(obs)
        action2 = controller.get_action(obs)

        assert np.allclose(action1, action2)

    def test_different_architectures(self):
        """Test controllers with different architectures"""
        # Small controller
        small = NeuralController(input_size=20, hidden_sizes=[16], output_size=4)
        assert small.num_params < 500

        # Large controller
        large = NeuralController(input_size=20, hidden_sizes=[128, 64, 32], output_size=4)
        assert large.num_params > 5000

        # Test both work
        obs = np.random.randn(20)
        small_action = small.get_action(obs)
        large_action = large.get_action(obs)

        assert small_action.shape == (1, 4)
        assert large_action.shape == (1, 4)

    def test_parameter_count(self):
        """Test parameter counting is correct"""
        controller = NeuralController(input_size=20, hidden_sizes=[32, 16], output_size=4)

        # Manual count
        # Layer 1: 20 -> 32: (20*32 + 32) = 672
        # Layer 2: 32 -> 16: (32*16 + 16) = 528
        # Layer 3: 16 -> 4: (16*4 + 4) = 68
        # Total: 672 + 528 + 68 = 1268

        expected_params = (20 * 32 + 32) + (32 * 16 + 16) + (16 * 4 + 4)
        assert controller.num_params == expected_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
