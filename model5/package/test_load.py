"""
Example usage of loading a model from the package in model 5.

To run this script from the repo root directory, use:
python -m model5.package.test_load
"""

import os
import sys
import pprint

# Import util.preprocessing. If import fails, add root path to sys.path
try:
    from model5.package import load_saved_model
except Exception:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from model5.package import load_saved_model

def main():    
    # Model saving directory (set to None if you don't want to save models)
    save_dir = "model5/saved_models"
    
    # Load saved model example
    if save_dir is not None:
        loaded_model_info = load_saved_model(save_dir)
        pprint.pprint(loaded_model_info)

if __name__ == '__main__':
    main()