import os
import shutil
import subprocess
import pytest


# Clean up: delete logs from previous test
for i in range(1001, 1005):
    log_folder = "logs/exp_"+str(i)
    if os.path.isdir(log_folder):
        shutil.rmtree(log_folder)


def test_1():
    
    args = [
        "--exp-id", 1001,
        "--algo", "a2c",
        "--env", 'widowx_reacher-v1',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_2():
    
    args = [
        "--exp-id", 1002,
        "--algo", "a2c",
        "--env", 'widowx_reacher-v3',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_3():
    
    args = [
        "--exp-id", 1003,
        "--algo", "her",
        "--env", 'widowx_reacher-v2',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_4():
    
    args = [
        "--exp-id", 1004,
        "--algo", "her",
        "--env", 'widowx_reacher-v4',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0
