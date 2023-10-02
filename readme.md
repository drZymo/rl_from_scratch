# Reinforcement Learning from Scratch

These are the exercises for the course "Reinforcement Learning from Scratch" that I give at [Sioux Technologies](https://www.sioux.eu/).

# Installation

If you want to run these exercises on your own computer you have to install the following packages.

1. Install an Anaconda environment like Miniconda (https://docs.conda.io/en/latest/miniconda.html) for Python 3.x.
2. Open an "Anaconda prompt" and type the following command to create a new python environment.

       conda create -n rl_from_scratch python==3.11.* pytorch torchvision torchaudio -c pytorch

   Or if you have an NVIDIA GPU:

       conda create -n rl_from_scratch python==3.11.* pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

3. Activate the environment.

       conda activate rl_from_scratch

4. Install additional packages.

       conda install ipython ipykernel swig
       pip install stable_baselines3[extra]
       pip install gymnasium[box2d]

5. Open this repository with Visual Studio Code and select `rl_from_scratch` as the Python environment.
6. Have fun
