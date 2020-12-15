import os
import subprocess
import pytest


def test_1():
    
    args = [
        '--exp-id', 1001,
        '--n-eval-steps', 100,
        '--log-info', 1,
        '--plot-dim', 0,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_2():
    
    args = [
        '--exp-id', 1002,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 2,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_3():
    
    args = [
        '--exp-id', 1003,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 3,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_4():
    
    args = [
        '--exp-id', 1004,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 0,
        '--render', 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0
