
# Import necessory libraries for setting up project template

import os
import logging
from pathlib import Path

# Set up information level log for tracking progress
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Set up Project name
project_name = "chest_cancer_classifier"

# Files that i need for this project
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/experiments.ipynb",
    "templates/index.html",
    "static/style.css",
    "app.py"
]


# Iterate over each file path in the list of files
for filepath in list_of_files:
    # Convert the filepath string to a Path object for easier manipulation
    filepath = Path(filepath)
    
    # Split the filepath into its directory and filename components
    filedir, filename = os.path.split(filepath)

    # Check if the directory part of the filepath is not empty
    if filedir != "":
        # Create the directory if it doesn't exist, without raising an error if it does
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Check if the file does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Open the file in write mode (this creates the file if it doesn't exist)
        with open(filepath, "w") as f:
            pass  # No content is written to the file
            logging.info(f"Creating empty file: {filepath}")

    else:
        # Log that the file already exists
        logging.info(f"{filename} already exists.")