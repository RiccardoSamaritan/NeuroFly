# NeuroFly

Neuroevolution for quadcopter drone control using Evolution Strategies and CMA-ES.

## Overview

Instead of backpropagation, NeuroFly evolves neural network controllers:
1. Create population of random controllers
2. Evaluate each by flying in simulator
3. Keep the best, mutate to create next generation
4. Repeat until convergence

**Goal**: Achieve stable hovering with neuroevolution.

## Simulator

This project includes **gym-pybullet-drones** locally for simplicity.

- **Source**: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- **Environment**: CtrlAviary.py
- **License**: MIT

## References

1. Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., & Schoellig, A. P. (2021). *Learning to Fly—A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control*. In _2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ (pp. 7512–7519). [DOI](https://doi.org/10.1109/IROS51168.2021.9635857)

