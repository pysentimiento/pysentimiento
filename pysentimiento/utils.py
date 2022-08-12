"""
Auxiliary functions for pysentimiento.
"""
import json


def clean_key(k):
    return k.split("_", 1)[1]

def load_evaluation(path):
    """
    Load evaluation file generated from benchmarking with bin/train.py
    """
    with open(path) as f:
        model_evaluation = json.load(f)
        clean_evaluations = []
        for task in model_evaluation["evaluations"].keys():
            task_evaluations = model_evaluation["evaluations"][task]
            clean_evaluations = [
                {clean_key(k): v for k, v in ev.items()}
                for ev in task_evaluations
            ]

            model_evaluation["evaluations"][task] = clean_evaluations

        return model_evaluation