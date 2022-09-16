# User Allocation in Mobile Edge Computing: A Reinforcement Learning Approach

## Installation Information

Implementation is done using conda and jupyter-notebook. 

Install Anaconda in your system. Then create a new conda environment (using environment.yml) using the following:

```bash
conda env create -f environment.yml
conda activate mecrl
```

## Project Directory Structure

- `agent_tensorboard`: Tensorboard files while training the network
- `dataset`: YOLO and MobilenetV2 execution dataset
- `dataset_generator`: Generate Dataset for training RL agent
- `eua`: EUA dataset map, Australia CBD area and codes to analyze various EUA zones
- `allocation_results_published`: **Allocation Results** are stores here in .csv format for ICWS2021
- `plots`: Various plots genereated from the .csv files are stored here
- `training_plots`: Loss and rewards obtained while training the RL network is are stored here
- `trained_agents`: Trained RL agents for use

## Notebooks

- `allocation.ipynb`: Main program to obtain allocation with different algorithms. All the algorithms are executed via function calls. The program will generate a `.csv` file. Use the .csv file to obtain plots.
- `allocation_threat.ipynb`: Allocation results for different quantization factors and different training steps
- `plot_*.ipynb`: This program generates plot from `.csv` files. Allocation and agent training plots.
- `trained_agents/edge_agent_*.zip`: DRL agent stored after training. The agent generated for different thresholds (represented by thres10)
- `rl_training.ipynb`: Program to train the RL agent for various threshold.
- `rl_training_threats_act.ipynb`: Program to train the RL agent with various quantization size
- `rl_training_threats_und.ipynb`: Program to train the RL agent with low training steps (under trained agent)
- `dataset_generator/gen_*.py`: Codes are used to generate dataset by executing YOLO and Mobilenetv2
- `eua/users.csv` and `eua/servers.csv`: Location of Usera and Servers (EUA Dataset)
- `dataset_generator/workload`: GPU workload generator used in `gen_*.py`


Please use `allocation.ipynb` to obtain the desired results. All the blocks except last block are function definitions. The last block of this Jupyter Notebook contains `for` loop which varies number of users and servers configuration. Please change the loop accordingly. Moreover, you can change the filename of resulting files.


### Paper:
S. P. Panda, A. Banerjee and A. Bhattacharya, "User Allocation in Mobile Edge Computing: A Deep Reinforcement Learning Approach," 2021 IEEE International Conference on Web Services (ICWS), 2021, pp. 447-458, doi: 10.1109/ICWS53863.2021.00064.

> This is the code for our paper ["User Allocation in Mobile Edge Computing: A Reinforcement Learning Approach"](https://ieeexplore.ieee.org/document/9590334).
