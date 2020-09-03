# Reinforcement Learning from Scratch

These are the exercises for the course "Reinforcement Learning from Scratch" that I give at [Sioux Technologies](https://www.sioux.eu/).

# Installation

If you want to run these exercises on your own computer you have to install the following packages.

1. Install an Anaconda environment like Miniconda (https://docs.conda.io/en/latest/miniconda.html) for Python 3.7/3.8
2. Open an "Anaconda prompt" and type the following command to create a new python environment.

       conda create -n rl_from_scratch python==3.7.* jupyter matplotlib swig

3. Activate the environment so we can install additional packages.

       conda activate rl_from_scratch

4. Install the following packages with pip.

       pip install tensorflow gym[atari,box2d]

5. Go the folder where you checked out this repository and start the Jupyter notebook server

       jupyter notebook .

   This should open a new browser window where you can start playing around with the exercises.