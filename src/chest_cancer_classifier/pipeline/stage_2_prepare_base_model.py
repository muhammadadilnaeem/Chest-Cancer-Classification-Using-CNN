
from src.chest_cancer_classifier.config.configuration import ConfigurationManager
from src.chest_cancer_classifier.components.prepare_base_models import PrepareBaseModel
from src.chest_cancer_classifier import logger

# Define the name of the stage for logging and tracking purposes
STAGE_NAME = "Prepare base model"

# Class definition for the training pipeline related to preparing the base model
class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        # Constructor method for the class; currently does not perform any initialization
        pass

    def main(self):
        # Main method to orchestrate the base model preparation process
        # Retrieve configuration settings for preparing the base model
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        
        # Create an instance of PrepareBaseModel with the retrieved configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Call the method to get the base model
        prepare_base_model.get_base_model()
        
        # Call the method to update the base model
        prepare_base_model.update_base_model()

# Entry point of the script
if __name__ == '__main__':
    try:
        # Log the start of the stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the training pipeline and run the main method
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        
        # Log the completion of the stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        raise e  # Reraise the exception for further handling