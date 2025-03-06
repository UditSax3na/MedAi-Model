import os
from Constants import DEFAULT_PARENT_DIR
current_dir = os.path.abspath(__file__)  # Get the current file path
parent_dir = os.path.dirname(current_dir)  # Move up one directory
grandparent_dir = os.path.dirname(parent_dir)  # Move up two directories

print("Current Directory:", DEFAULT_PARENT_DIR)