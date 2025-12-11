# NeuroFly - NEAT Drone Control

A hybrid control system combining [NEAT](https://neat-python.readthedocs.io/en/latest/neat_overview.html) (NeuroEvolution of Augmenting Topologies) with PID control for autonomous drone stabilization.

## Overview

This project is part of the final exam for the *Optimization for Artificial Intelligence* course at University of Trieste
It explores whether evolutionary algorithms can learn optimal control strategies autonomously, without manual tuning of PID parameters. The system uses:

- **Neural Network** (evolved via NEAT): Takes position error and velocity as input (6 values) and outputs small position offsets (±0.2m)
- **PID Controller**: Takes the adjusted target position and computes the RPM values for the 4 motors
- **Evolutionary Training**: NEAT evolves networks that minimize distance from the target hover position [0, 0, 1]

### Why NEAT + PID?

While PID alone is sufficient for static hovering, this architecture provides:

1. **Learning-Based Approach**: Discovers control strategies automatically without manual PID tuning (Kp, Ki, Kd parameters)
2. **Scalability**: Foundation for complex tasks where PID alone isn't enough (dynamic trajectories, obstacle avoidance, environmental adaptation)
3. **Research Platform**: Compares classical control (PID) vs learned control (NEAT) to understand their trade-offs
4. **Extensibility**: Easy to switch to direct RPM control (ActionType.RPM) where NEAT becomes essential

The current hovering task serves as a **proof of concept** - once validated, the system can tackle more complex scenarios where evolutionary learning provides real advantages over pure PID control.

## Project Structure

```
NeuroFly/
├── src/
│   └── neat/
│       ├── drone_evaluator.py    # Genome evaluation in HoverAviary
│       └── train.py               # NEAT trainer with parallel evaluation
├── config-drone                   # NEAT configuration file
├── tests/                         # Test suite used for Test Driven Development
├── models/                        # Saved trained genomes in pickle format
├── train_drone.py                 # Main training script
├── test_genome.py                 # Test genome with GUI
```

## Setup

```bash
# Clone repository
git clone https://github.com/RiccardoSamaritan/NeuroFly.git

# Install dependencies
pip install -r requirements.txt
pip install -e gym-pybullet-drones/
```

## Usage

### 1. Training

Default training with 50 generations with all CPU cores:
```bash
python train_drone.py
```

Custom training:
```bash
python train_drone.py --generations 100 --workers 4 --output models/my_genome.pkl
```

Options:
- `--generations N`: Number of generations (default: 50)
- `--workers N`: Parallel workers (default: all CPU cores)
- `--checkpoint-interval N`: Save checkpoint every N generations (default: 10)
- `--config PATH`: NEAT config file (default: config-drone)
- `--output PATH`: Output genome file (default: best_genome.pkl)

### 4. Test Trained Genome

Test genome with [PyBullet](https://pybullet.org/wordpress/) GUI:
```bash
python test_genome.py best_genome.pkl
```

Run multiple episodes:
```bash
python test_genome.py best_genome.pkl --episodes 5
```

## Configuration

Edit `config-drone` to adjust NEAT parameters:
- **Population size**: `pop_size = 150`
- **Network structure**: `num_inputs = 6`, `num_outputs = 3`
- **Mutation rates**: `weight_mutate_rate`, `conn_add_prob`, etc.

## Architecture

### Environment
- **HoverAviary**: PyBullet-based drone simulation with PID control. It consists of a quadcopter that must hover at the target position [$x=0, y=0, z=1$] meters.

### Neural Network Input (6 values)
1. **Position Error** (3 values): `target_pos - drone_pos`
   - How far the drone is from [0, 0, 1]
2. **Drone Velocity** (3 values): Current velocity vector
   - How fast the drone is moving

### Neural Network Output (3 values)

- **Position Offsets**: Small adjustments (±0.2m) in $x, y, z$
- These offsets are added to the target position [0, 0, 1]
- The PID controller then stabilizes the drone at this adjusted target

### Control Flow
```
Environment State → Extract 6 features → Neural Network → Position Offsets
                                                ↓
                                         Adjusted Target = [0,0,1] + offsets
                                                ↓
                                         PID Controller → Motor RPMs
```

### NEAT Configuration
- **Inputs**: 6 (position error + velocity)
- **Outputs**: 3 (target position offsets for PID)
- **Activation**: tanh
- **Initial connections**: partial_direct 0.5
- **Network type**: Recurrent

### Fitness Function
```python
# Per-step fitness: inverse distance to target
dist_score = 1.0 / (1.0 + distance_to_target)
cumulative_fitness += dist_score

# Bonus for completing the full episode
if steps >= max_steps:
    cumulative_fitness += 50.0
```

The drone fails early if:
- Distance from target > 2.5m
- Altitude < 0.05m (crashed)

### Parallelization

The training script uses `neat.ParallelEvaluator` to evaluate genomes concurrently, speeding up the evolutionary process on multi-core systems. 

To adjust the number of parallel workers, use the `--workers` command-line argument when running `train_drone.py`.

## Development

Project follows Test-Driven Development (TDD):
1. Write tests first
2. Implement features
3. Verify all tests pass

To ensure TDD compliance, GitHub Actions were used to run tests on each commit.

## References

1. Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., & Schoellig, A. P. (2021). *Learning to Fly—A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control*. In _2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ (pp. 7512–7519). [DOI](https://doi.org/10.1109/IROS51168.2021.9635857)