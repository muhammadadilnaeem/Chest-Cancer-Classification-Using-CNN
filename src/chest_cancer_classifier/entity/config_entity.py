

from dataclasses import dataclass  # Import the dataclass decorator for creating data classes
from pathlib import Path  # Import Path for easy file path handling

# Configuration class for data ingestion settings
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path  # Directory where data ingestion artifacts will be stored
    source_URL: str  # URL to the source dataset
    local_data_file: Path  # Path for the downloaded local data file
    unzip_dir: Path  # Directory where the data will be unzipped

# Configuration class for preparing the base model settings
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path  # Directory for storing base model artifacts
    base_model_path: Path  # Path to the original base model file
    updated_base_model_path: Path  # Path to the modified base model file
    params_image_size: list  # List containing image dimensions (e.g., [height, width, channels])
    params_learning_rate: float  # Learning rate for training the model
    params_include_top: bool  # Flag to include the top layer of the model
    params_weights: str  # Source of pre-trained weights (e.g., 'imagenet')
    params_classes: int  # Number of classes for classification

# Configuration class for training settings
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path  # Directory for storing training artifacts
    trained_model_path: Path  # Path where the trained model will be saved
    updated_base_model_path: Path  # Path to the updated base model
    training_data: Path  # Path to the training dataset
    params_epochs: int  # Number of epochs for training
    params_batch_size: int  # Batch size for training
    params_is_augmentation: bool  # Flag to indicate if data augmentation is used
    params_image_size: list  # Image dimensions for input to the model

# Configuration class for evaluation settings
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path  # Path to the model to be evaluated
    training_data: Path  # Path to the training dataset used for evaluation
    validation_data_dir: Path  # Path to the validation dataset directory
    all_params: dict  # Dictionary containing all relevant parameters for evaluation
    mlflow_uri: str  # URI for MLflow tracking server
    params_image_size: list  # Image dimensions for input to the model
    params_batch_size: int  # Batch size for evaluation