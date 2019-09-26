"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from tennis.train import make_plot, setup, train
import os
import platform
from unityagents import UnityEnvironment

# !!!!!!!!! YOU MAY NEED TO EDIT THIS !!!!!!!!!!!!!!!
if platform.system() == 'Windows':
    print("Loading Windows x86 64-bit Tennis environment.")
    env = UnityEnvironment(file_name=r"Tennis_Windows_x86_64\Tennis.exe")
elif platform.system() == 'Linux':
    print("Loading Linux Tennis environment.")
    env = UnityEnvironment(file_name=r"Tennis_Linux\Tennis.exe")
else:
    print("Only Windows and Linux supported.")
    raise RuntimeError

if __name__ == "__main__":

    # Delete the old output files
    # -----------------------------------------------------------------------------------
    for name in ['score.png', 'scores.npz', 'checkpoint.pth']:
        if os.path.isfile(name):
            os.remove(name)

    # Setup the environment and agent
    # -----------------------------------------------------------------------------------
    agent = setup(env)

    # Perform the training
    # -----------------------------------------------------------------------------------
    print('Training the agent.')
    train(agent=agent, env=env)

    # Make some pretty plots
    # -----------------------------------------------------------------------------------
    print('Make training plot called rewards.png.')
    make_plot()
