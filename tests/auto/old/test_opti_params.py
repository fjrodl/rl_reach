import os
import shutil
import subprocess
import pytest


# # Clean up: delete logs from previous test
log_folder = "logs/test/opti/"
if os.path.isdir(log_folder):
    shutil.rmtree(log_folder)


def test_opti():
    
    args = [
        "-optimize",
        "--algo", "ppo",
        "--env", 'widowx_reacher-v1',
        "--n-timesteps", 2000,
        "--n-trials", 2,
        "--n-jobs", 8,
        "--sampler", "tpe",
        "--pruner", "median",
        "--n-startup-trials", 1,
        "--n-evaluations", 5,
        "--log-folder", log_folder
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0
