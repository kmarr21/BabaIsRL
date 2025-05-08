# BabaIsRL: Enhancing Deep Q-Learning with Domain Heuristics and a Neurosymbolic Decision Tree Advisor

This repository contains the implementation of a Deep Q-Learning agent enhanced with domain-specific knowledge and a neurosymbolic integration approach for solving the KeyCorridors environment: a custom grid-based planning task with Last-In-First-Out (LIFO) key-door constraints.

## Project Overview

The KeyCorridors environment is a 6×6 gridworld where a bunny agent must collect keys and open doors to reach a goal state. The environment includes:
- Two collectible keys per level, each associated with a color
- Two doors, one per color, which can only be opened with the matching key
- A LIFO key usage constraint; only the most recently collected key may be used to open a door
- Enemy robots with predictable movement patterns that must be avoided
- Impassable walls that enforce path-planning and corridor constraints

The project implements and evaluates three key enhancements to a baseline DQN:
1. **Enhanced DQN**: Incorporates reward shaping and state augmentation
2. **KSM-Enhanced DQN**: Adds a Key Strategy Metric (KSM) to guide key collection order
3. **Neurosymbolic Decision Tree DQN (NDT-DQN)**: Integrates a handcrafted symbolic decision tree with a neural policy

Each enhancement is evaluated across six different level templates of varying complexity.

## Code Structure

### Main Agent Files
- `dqn_agent_enhanced.py`: Implementation of DQN with reward shaping and state augmentation (each able to be flagged on and off)
- `neurosymbolic_agent.py`: Implementation of the Neurosymbolic Decision Tree DQN
- `neural_symbolic_tree.py`: The symbolic decision tree component for the NDT-DQN
- `prioritized_replay_buffer.py`: Implementation of prioritized experience replay

### Environment Files
- `template_lifo_corridors.py`: The custom KeyCorridors environment implementation

### Training Scripts
- `train_enhanced_dqn.py`: Script for training the enhanced DQN
- `train_neuro_symbolic.py`: Script for training the NDT-DQN
- `calculate_ksm_factors.py`: Implementation of the Key Strategy Metric to test if values are as expected
- `validate_ksm_logic.py`: Utility to validate KSM calculations

### Experiment Scripts
- `run_experiment1_6k.sh`: Run Experiment 1 (ablation study with 6000 episodes)
- `run_experiment2_6k.sh`: Run Experiment 2 (KSM comparison with 6000 episodes)
- `run_neuro_symbolic.sh`: Run Experiment 3 (NDT-DQN with 5000 episodes)
- `experiment_runner_6k.sh`: Helper script for Experiment 1
- `experiment_runner_neuro.sh`: Helper script for Experiment 3
- `monitor_experiments.sh`: Script to monitor experiment progress
- `monitor_neuro_symbolic.sh`: Script to monitor NDT-DQN experiment progress

### Plotting Scripts
- `plot_6k_experiments.py`: Generate plots for Experiment 1 results
- `plot_neurosymbolic_experiments.py`: Generate plots for Experiment 3 results
- `new_plot_results.py`: General plotting utility
- `run_6k_plots.sh`: Script to run Experiment 1 plots
- `run_neurosymbolic_plots.sh`: Script to run Experiment 3 plots
- `generate_plots.sh`: General plotting script
- `generate_neuro_symbolic_plots.sh`: Helper for neurosymbolic plots

## Setup and Installation

This project was developed and tested on a Google Cloud Platform (GCP) virtual machine with the following configuration:
- Instance type: n1-standard-4
- GPU: NVIDIA T4
- Disk: Balanced persistent disk, 100 GB
- Image: Deep Learning VM for PyTorch 1.12 with CUDA 11.3 (M112)

### Dependencies

The code requires the following Python packages:
- PyTorch (version 1.12)
- NumPy
- Matplotlib
- tqdm

You can install the required dependencies using pip:

```bash
pip install torch==1.12.0 numpy matplotlib tqdm
```

## Running the Experiments

Clone the repository and navigate to the BabaIsRL directory:

```bash
git clone https://github.com/kmarr21/BabaIsRL.git
cd BabaIsRL
```

### Experiment 1: Ablation Study

This experiment compares four configurations: baseline DQN, DQN with reward shaping only, DQN with state augmentation only, and enhanced DQN with both features.

**Run the experiment** (from inside the BabaIsRL directory):
```bash
# Make the script executable if needed
chmod +x run_experiment1_6k.sh
# Run the experiment
./run_experiment1_6k.sh
```

**Monitor the progress** (from inside the BabaIsRL directory):
```bash
chmod +x monitor_experiments.sh
./monitor_experiments.sh
```

### Experiment 2: KSM Comparison

This experiment compares three configurations: enhanced DQN without KSM, enhanced DQN with standard KSM, and enhanced DQN with adaptive KSM.

**Run the experiment** (from inside the BabaIsRL directory):
```bash
# Make the script executable if needed
chmod +x run_experiment2_6k.sh
# Run the experiment
./run_experiment2_6k.sh
```

**Monitor the progress** (from inside the BabaIsRL directory):
```bash
chmod +x monitor_experiments.sh
./monitor_experiments.sh
```

### Experiment 3: NDT-DQN Evaluation

This experiment compares the enhanced DQN with the Neurosymbolic Decision Tree DQN.

**Run the experiment** (from inside the BabaIsRL directory):
```bash
# Make the script executable if needed
chmod +x run_neuro_symbolic.sh
# Run the experiment
./run_neuro_symbolic.sh
```

**Monitor the progress** (from inside the BabaIsRL directory):
```bash
chmod +x monitor_neuro_symbolic.sh
./monitor_neuro_symbolic.sh
```

### Generating Plots

There are two methods for generating plots from the experiment results:

#### Method 1: Using `run_6k_plots.sh` (for Experiment 1 and 2)

This script should be run from your home directory, as it expects the plotting script to be in your home directory:

1. First, copy the plotting script to your home directory:
```bash
cp BabaIsRL/plot_6k_experiments.py ~/
```

2. Then run the plotting script (from your home directory):
```bash
cp BabaIsRL/run_6k_plots.sh ~/
chmod +x ~/run_6k_plots.sh
~/run_6k_plots.sh
```

This will create a timestamped directory in your home folder containing all plots.

#### Method 2: Using `generate_plots.sh` (General method)

This is a more general plotting script that can be used for any experiment:

1. Copy the script to your home directory:
```bash
cp BabaIsRL/new_plot_results.py ~/
cp BabaIsRL/generate_plots.sh ~/
```

2. Run the script (from your home directory):
```bash
chmod +x ~/generate_plots.sh
~/generate_plots.sh experiment1 experiment2  # Or specify which experiment to plot
```

This will generate plots in a `plots/` directory in your current working directory.

#### For Experiment 3 (NDT-DQN):

To generate plots for the neurosymbolic experiment:
```bash
cp BabaIsRL/plot_neurosymbolic_experiments.py ~/
cp BabaIsRL/run_neurosymbolic_plots.sh ~/
chmod +x ~/run_neurosymbolic_plots.sh
~/run_neurosymbolic_plots.sh
```

## Results

The experimental results show that:

1. State augmentation consistently outperforms reward shaping, with a synergistic effect when both are combined.
2. The Key Strategy Metric (KSM) provides substantial benefits in environments where key ordering is critical.
3. The NDT-DQN demonstrates dramatically faster learning across most environments, typically reducing training time by 30-50%.

Detailed results and analysis can be found in the project report.

## Environment Templates

The agents are evaluated across six level templates of increasing complexity:
- **Basic**: A simple open structure with minimal constraints
- **Corridors**: Long vertical and horizontal corridors with enemy patrols
- **Sparse**: Widely separated keys and doors emphasizing planning
- **Zipper**: A deceptive layout where the key closer to the agent is not the optimal first choice
- **Bottleneck Med**: A moderate bottleneck layout with a narrow central gap
- **Bottleneck Hard**: A more constrained variant requiring longer navigation paths

## License

This project is licensed under a custom license that requires explicit written permission for any use, copying, modification, distribution, or other utilization of the code and associated materials. See the LICENSE.md file for complete details.

## Acknowledgements

This project was developed as part of research on integrating domain-structured and symbolic reasoning with deep reinforcement learning.

The KeyCorridors environment was loosely inspired by the puzzle game "Baba Is You" by Hempuli (Arvi Teikari) (https://hempuli.com/baba/), an innovative puzzle game where rules themselves are objects that can be manipulated. The main similarity is that both involve a bunny agent seeking to unlock doors. Luckily, this bunny has to deal with a much simpler world, though it does have the added challenge of a LIFO constraint.