# DRL applied to "Tennis" Environment 
This repo trains a Deep Reinforcement Learning (DRL) agent to solve the Unity ML-Agents "Tennis" environment. 
The motivation for this program was the 3rd project in the Udacity Deep Reinforcement Learning 
[Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

## Project Details
### Tennis Environment
The [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 
environment provided by [Unity](https://unity3d.com/machine-learning/) has two agents that control rackets to bounce a 
ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the 
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball 
in play.   
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each 
agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward 
(or away from) the net, and jumping.  

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

### Solving the Environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each 
agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.  

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started 
This repo is setup for a 64-bit Windows 10 Home or AWS environment.  If you prefer another os, please see the source 
Udacity [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).
Note to determine if your Windows is 64-bit or 32-bit follow this 
[link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64).
If you install one of the other environments, the following operations should be the same.  Only the Tennis environment
downloaded into the `p3_collab-compet` folder will be different. 

### Python 'drlnd' Virtual Environment (venv)

To set up your python environment to run the code in this repository, follow the instructions below.  Note if you have 
already created the 'drlnd' venv for other Udacity Deep Reinforcement Learning NanoDegree (DRLND), you can simply 
activate it. 

1. If not already installed, install the Anaconda Python distribution from [here](https://www.anaconda.com/distribution/). 

2. Create (and activate) a new environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Ensure you have the "Build Tools for Visual Studio 2019" installed from this 
[site](https://visualstudio.microsoft.com/downloads/).  
This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also 
be very helpful.  This was confirmed to work in Windows 10 Home.  

4. Follow the instructions in this [repo](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
5. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several 
dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

## Instructions
### Training the Agent
The agent is trained and tested with the `run_test.py` script, which can be executed in your favorite IDE of the 
following command:

    python run_test.py 

### Pretty plot
The raw scores are save as `scores.npz` and can be plotted and saved as `scores.png` with the command:

    python plot.py  

### Saved model Weights
The trained model is saved as `checkpoint.pth` in the root directory and can be loaded with the command:  

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
### Report
A [report](Report.md) providing a description of the implementation, a plot of the rewards and ideas for future work can
be found in `Report.md`.

## License
This code is copyright under the [MIT License](LICENSE).

## Contributions
Please feel free to raise issues against this repo if you have any questions or suggestions for improvement.
