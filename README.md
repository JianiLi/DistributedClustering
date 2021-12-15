# Source code for Submission 32

- Tested on python 3.5
- Install necessary packages: 
    - python -m pip install -r requirements.txt

## 1. TargetLocalization - Section 5.1

The task is to estimate the location of the target by minimizing the squared error loss of noisy streaming sensor data. 
We consider a network of 100 agents with four targets.
Agents in the same color share the same target, however, they do not know this group information beforehand.

### Instructions

- Run main.py to reproduce the results shown in the paper.
	- In main.py, we simulate four cases: "no-cooperation", "loss", "distance", " average", as explained in the paper.
	- "numAgents" is the total number of agents in the network.
- Run plotWeightMatrix to generate weight matrix (Fig.3 in the paper).


## 2. Digit Classification - Section 5.2

The task is to classify the digits into ten classes from 0-9.

### Dataset
- MNIST
- Synthetic digits with random embedded background.

### Instructions

- Run main.py to reproduce the results shown in the paper.
	- In main.py, we simulate four cases: "no-cooperation", "loss", "distance", " average", as explained in the paper.
	- "numAgents" is the total number of agents in the network.
- Run plot.py to generate Fig.7 in the paper.
- Run plotWeightMatrix to generate weight matrix (Fig.6 in the paper).




