
# LIBRARIES IMPORT
from src.chest_cancer_classifier.config.configuration import ConfigurationManager
from src.chest_cancer_classifier.components.model_trainer import Training
from src.chest_cancer_classifier import logger

# Define the name of the stage for logging and tracking purposes
STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Main method to orchestrate the model training process
        # Retrieve configuration settings for model training
        config = ConfigurationManager()
        training_config = config.get_training_config()
        
        # Create an instance of Training with the retrieved configuration
        training = Training(config=training_config)

        # Call the method to get the base model
        training.get_base_model()
        
        # Call the method to prepare the training and validation generators
        training.train_valid_generator()
        # Call the method to train the model
        training.train()

# Entry point of the script
if __name__ == '__main__':
    # Main method
    try:
        # Log the start of the stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # Create an instance of ModelTrainingPipeline and run the main method
        obj = ModelTrainingPipeline()
        # Execute the main process of the model training pipeline
        obj.main()
        # Log the successful completion of the stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    # Log any exceptions that occur during the execution
    except Exception as e:
        # Reraise the exception for further handling
        logger.exception(e)
        # Reraise the exception
        raise e
