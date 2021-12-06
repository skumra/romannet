# RoManNet

This repository contains the implementation of the Robotic Manipulation Network (RoManNet) which is a vision-based model architecture to learn the action-value functions and predict manipulation action candidates. More details can be found in the following paper:

#### Learning Robotic Manipulation Tasks via Task Progress Based Gaussian Reward and Loss Adjusted Exploration

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[Xplore](https://ieeexplore.ieee.org/abstract/document/9625869) | [arxiv](https://arxiv.org/abs/2103.01434)

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:
```
@article{kumra2022learning, 
author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat}, 
journal={IEEE Robotics and Automation Letters},  
title={Learning Robotic Manipulation Tasks via Task Progress Based Gaussian Reward and Loss Adjusted Exploration},   
year={2022},  
volume={7},  
number={1},  
pages={534-541},  
doi={10.1109/LRA.2021.3129833}
}
```

## Installation
- Checkout the repository
```bash
$ git clone https://github.com/skumra/romannet.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd romannet
$ pip install -r requirements.txt
```

- Install the [CoppeliaSim](http://www.coppeliarobotics.com/) simulation environment


## Usage
- Run CoppeliaSim (navigate to your CoppeliaSim directory and run `./sim.sh`). From the main menu, select `File` > `Open scene`..., and open the file `romannet/simulation/simulation.ttt` from this repository.

- In another terminal window, run the following:
```bash
python main.py <optional args>
```

Note: Various training/testing options can be modified or toggled on/off with different flags (run `python main.py -h` to see all options)

### Acknowledgement
Some parts of the code and simulation environment has been borrowed from [andyzeng/visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping) for fair comparison of our work in simulation.
