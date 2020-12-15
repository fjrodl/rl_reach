import os
import shutil
import subprocess
import pytest


# # Clean up: delete logs from previous test
log_folder = "logs/test/ppo/"
if os.path.isdir(log_folder):
    shutil.rmtree(log_folder)


def test_train():
    
    args = [
        "-n", 100000,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", log_folder
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0


def test_enjoy():
    
    args = [
        "-n", 300,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", log_folder,
        "--render", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'enjoy.py'] + args)

    assert return_code == 0

