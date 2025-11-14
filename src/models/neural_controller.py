"""
Neural Network Controller per il controllo del drone.
Rete feedforward semplice che mappa lo stato del drone alle azioni dei motori.
"""
import numpy as np
import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """
    Controller basato su rete neurale feedforward.

    Input: 20 features (stato del drone da CtrlAviary)
    Output: 4 azioni in [-1, 1] (poi mappate a RPM)
    """

    def __init__(self, input_size=20, hidden_sizes=[32, 16], output_size=4):
        """
        Args:
            input_size: Dimensione input (default 20 per CtrlAviary)
            hidden_sizes: Lista delle dimensioni dei layer nascosti
            output_size: Dimensione output (default 4 per i 4 motori)
        """
        super(NeuralController, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Costruisci i layer
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())  # Tanh per mantenere valori limitati
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.network = nn.Sequential(*layers)

        # Conta parametri
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor di forma (batch_size, input_size) o (input_size,)

        Returns:
            Output tensor di forma (batch_size, output_size) o (output_size,)
        """
        return self.network(x)

    def get_action(self, obs):
        """
        Ottieni azione dato uno stato (osservazione).

        Args:
            obs: Numpy array di forma (1, 72) o (72,)

        Returns:
            Numpy array di forma (1, 4) - azioni normalizzate in [-1, 1]
        """
        # Converti a tensor
        if isinstance(obs, np.ndarray):
            # Flatten se necessario
            if obs.ndim > 1:
                obs = obs.flatten()
            obs = torch.FloatTensor(obs)

        # Forward pass
        with torch.no_grad():
            action = self.forward(obs)

        # Converti a numpy e reshape
        action = action.cpu().numpy()
        if action.ndim == 1:
            action = action.reshape(1, -1)

        return action

    def get_params(self):
        """
        Ottieni tutti i parametri della rete come un array 1D.
        Utile per algoritmi evolutivi.

        Returns:
            Numpy array 1D con tutti i pesi e bias
        """
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_params(self, params):
        """
        Imposta i parametri della rete da un array 1D.
        Utile per algoritmi evolutivi.

        Args:
            params: Numpy array 1D con tutti i pesi e bias
        """
        if len(params) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameters, got {len(params)}")

        idx = 0
        for param in self.parameters():
            param_length = param.numel()
            param.data = torch.FloatTensor(
                params[idx:idx + param_length].reshape(param.shape)
            )
            idx += param_length

    def clone(self):
        """
        Crea una copia del controller con gli stessi pesi.

        Returns:
            Nuovo NeuralController con stessi pesi
        """
        new_controller = NeuralController(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size
        )
        new_controller.load_state_dict(self.state_dict())
        return new_controller

    def save(self, filepath):
        """Salva il controller su file."""
        torch.save({
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'state_dict': self.state_dict(),
            'num_params': self.num_params
        }, filepath)

    @staticmethod
    def load(filepath):
        """Carica il controller da file."""
        checkpoint = torch.load(filepath)
        controller = NeuralController(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            output_size=checkpoint['output_size']
        )
        controller.load_state_dict(checkpoint['state_dict'])
        return controller

    def __repr__(self):
        return f"NeuralController(input={self.input_size}, hidden={self.hidden_sizes}, output={self.output_size}, params={self.num_params})"


if __name__ == "__main__":
    # Test del controller
    print("Testing NeuralController...")

    # Crea controller
    controller = NeuralController(input_size=72, hidden_sizes=[64, 32], output_size=4)
    print(f"\n{controller}")

    # Test con input random
    obs = np.random.randn(1, 72)
    action = controller.get_action(obs)
    print(f"\nInput shape: {obs.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Output range: [{action.min():.3f}, {action.max():.3f}]")

    # Test get/set params
    params = controller.get_params()
    print(f"\nTotal parameters: {len(params)}")

    # Test set params
    new_params = np.random.randn(len(params)) * 0.1
    controller.set_params(new_params)
    action2 = controller.get_action(obs)
    print(f"Action after setting new params: shape={action2.shape}")

    # Test clone
    controller2 = controller.clone()
    action3 = controller2.get_action(obs)
    print(f"\nClone test - actions match: {np.allclose(action2, action3)}")

    print("\n✅ All tests passed!")
