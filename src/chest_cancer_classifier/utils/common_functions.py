
# Import libraries
import os
import json
import yaml
import joblib
import base64
from typing import Any
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from chest_cancer_classifier import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: If any other error occurs during file reading.

    Returns:
        ConfigBox: Parsed content from YAML as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.error(f"YAML file '{path_to_yaml}' is empty")
        raise ValueError("YAML file is empty")
    except Exception as e:
        logger.exception(f"Error loading YAML file '{path_to_yaml}': {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): If True, logs creation for each directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): Path where JSON file will be saved.
        data (dict): Data to save in JSON format.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads JSON data from a file and returns it as a ConfigBox object.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data loaded from JSON file as a ConfigBox.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data in binary format using joblib.

    Args:
        data (Any): Data to save as binary.
        path (Path): Path where binary file will be saved.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads binary data using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Object stored in the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Gets the file size in kilobytes (KB).

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB, rounded to the nearest integer.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    logger.info(f"Size of '{path}': ~{size_in_kb} KB")
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring: str, fileName: str):
    """
    Decodes a base64 string and saves it as an image file.

    Args:
        imgstring (str): Base64-encoded image string.
        fileName (str): Path where the decoded image will be saved.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
    logger.info(f"Image decoded and saved to '{fileName}'")


def encodeImageIntoBase64(croppedImagePath: str) -> str:
    """
    Encodes an image file as a base64 string.

    Args:
        croppedImagePath (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image file.
    """
    with open(croppedImagePath, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    logger.info(f"Image at '{croppedImagePath}' encoded to base64")
    return encoded_string.decode('utf-8')
