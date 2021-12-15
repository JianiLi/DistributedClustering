
### Case Study - Target Localization
The task is to estimate the location of the target by minimizing the squared error loss of noisy streaming sensor data. 
We consider a network of 100 agents with four targets.
Agents in the same color share the same target, however, they do not know this group information beforehand.

### Dataset
- Data is generated in the code.

### Instructions
Tested on python 3.5

- Run main.py to reproduce the results shown in the paper.
	- In main.py, we simulate four cases: "no-cooperation", "loss", "distance", " average", as explained in the paper.
	- "numAgents" is the total number of agents in the network.
- Run plotWeightMatrix to generate weight matrix (Fig.3 in the paper).
