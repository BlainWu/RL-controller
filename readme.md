# Robustness of Reinforcement Learning

## 0. Introduction

This is a project for studying the robustness of reinforcement learning controllers in classical control problems, for 
instance the Cart-Pole problem.  

## 1. Initialising the environment

The reinforcement learning environment for this project is [Gym](https://www.gymlibrary.dev/), and the deep learning
platform is Pytorch.  

Install requirements packages:  
``pip install -r requirements.txt``

## 2. Running a Demo
### 2.1 LQR controller
Running the following command:  
``python LQR_Visible.py``  
this demo demonstrates the LQR controller given a position set-point of one meter when the time count is larger than 200.  

### 2.2 REINFORCE controller  
Running the following command:  
``python Discrete_RL_Visible.py``  

this demo demonstrates the REINFORCE controller given a position set-point of one meter when the time count is larger than 200.  
As the action space is discrete, this demo is applicabla to DQN controller as long as the model is loaded.  
### 2.3 DDPG controller
Running the following command:  
``python Continuous_RL_Visible.py``  
this demo demonstrates the DDPG controller whose action space is continuous time, given a position set-point of one meter when the time count is larger than 200.  