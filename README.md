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

```bibtex
@INPROCEEDINGS{panerati2021learning,
  title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control},
  author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021},
  pages={7512-7519},
  doi={10.1109/IROS51168.2021.9635857}
}
```
