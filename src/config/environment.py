import os
import json

def setup_env(file_path):
    with open(file_path, 'r') as f:
        env_vars = json.load(f)

    for key, value in env_vars.items():
        os.environ[key] = value
