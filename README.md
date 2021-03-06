# rl-navigation

## Introduction

This project was made as a part of the Deep Reinforcement Learning Nanodegree. 
A DQN agent is trained to navigate an environment and collect bananas. Unity environment is used for training. 
The code is written in Python 3 and Pytorch.

![Agent](resources/banana.gif)

## Installation
- First follow the instructions on the drlnd github page to setup the required packages and modules: [prerequisites](https://github.com/udacity/deep-reinforcement-learning/#dependencies).
- For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

    Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


- (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/15056/windows-32-64-bit-faq) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

- (_For AWS_) If you'd like to train the agent on AWS (and have not enabled a [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

## Environment

- The environment provides a reward of +1 is provided for collecting a yellow banana, and a reward of -1 for collecting a blue banana. 
- The goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.
- The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 
- Given this information, the agent has to learn how to best select actions. 
- Four discrete actions are available, corresponding to:

    - **`0`** - move forward.
    - **`1`** - move backward.
    - **`2`** - turn left.
    - **`3`** - turn right.

- The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

    