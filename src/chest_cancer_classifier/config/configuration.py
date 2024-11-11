
# Import necessary modules and classes from the chest_cancer_classifier package
from src.chest_cancer_classifier import *  # Import all components from the chest_cancer_classifier module
from src.chest_cancer_classifier.entity.config_entity import (DataIngestionConfig,
                                                              PrepareBaseModelConfig,
                                                              TrainingConfig,
                                                              EvaluationConfig)

from src.chest_cancer_classifier.utils.common_functions import read_yaml, create_directories  # Import utility functions for reading YAML files and creating directories

from src.chest_cancer_classifier.constants import *  # Import all constants defined in the constants module
import os  # Import the os module

# ConfigurationManager class to manage configuration settings for the project
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,  # Path to the main configuration file, defaulting to CONFIG_FILE_PATH
        params_filepath=PARAMS_FILE_PATH):  # Path to the parameters file, defaulting to PARAMS_FILE_PATH

        # Read the configuration and parameters from the specified YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create necessary directories as specified in the configuration
        create_directories([self.config.artifacts_root])

    # Method to retrieve the data ingestion configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Access the data ingestion configuration from the loaded config
        config = self.config.data_ingestion

        # Create the directory for data ingestion as specified in the configuration
        create_directories([config.root_dir])

        # Instantiate the DataIngestionConfig object with the relevant parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,  # Set the root directory
            source_URL=config.source_URL,  # Set the source URL for data ingestion
            local_data_file=config.local_data_file,  # Set the local path for the downloaded data file
            unzip_dir=config.unzip_dir  # Set the directory where the data will be unzipped
        )

        # Return the configured DataIngestionConfig object
        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        # Retrieve the configuration for preparing the base model
        config = self.config.prepare_base_model
        
        # Create necessary directories for the base model
        create_directories([config.root_dir])

        # Initialize the PrepareBaseModelConfig with relevant parameters
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        # Return the configuration object
        return prepare_base_model_config


    def get_training_config(self) -> TrainingConfig:
        # Retrieve training and base model configurations
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        # Define the path to the training data
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")

        # Create necessary directories for training
        create_directories([
            Path(training.root_dir)
        ])

        # Initialize the TrainingConfig with relevant parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        # Return the training configuration object
        return training_config


    def get_evaluation_config(self) -> EvaluationConfig:
        # Initialize the EvaluationConfig with relevant parameters
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Chest-CT-Scan-data",
            mlflow_uri="https://dagshub.com/muhammadadilnaeem/Chest-Cancer-Classification-Using-MLflow-and-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        # Return the evaluation configuration object
        return eval_config
    
    def get_training_config(self) -> TrainingConfig:
        # Retrieve training configuration details from the loaded config
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        # Define the path to the training data
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
        
        # Create the root directory for training if it does not exist
        create_directories([
            Path(training.root_dir)
        ])

        # Create a TrainingConfig object with the relevant parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),  # Root directory for training
            trained_model_path=Path(training.trained_model_path),  # Path to save the trained model
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),  # Path for the updated base model
            training_data=Path(training_data),  # Path to the training data
            params_epochs=params.EPOCHS,  # Number of epochs for training
            params_batch_size=params.BATCH_SIZE,  # Batch size for training
            params_is_augmentation=params.AUGMENTATION,  # Whether to use data augmentation
            params_image_size=params.IMAGE_SIZE  # Size of the images for training
        )

        # Return the constructed training configuration
        return training_config