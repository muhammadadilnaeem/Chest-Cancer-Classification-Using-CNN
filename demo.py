from src.chest_cancer_classifier import logger
from chest_cancer_classifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from chest_cancer_classifier.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline

# Define the name of the current stage in the data processing pipeline
STAGE_NAME = "Data Ingestion stage"

try:
        # Log the start of the data ingestion stage
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # Create an instance of the DataIngestionTrainingPipeline
        obj = DataIngestionTrainingPipeline()
        # Execute the main process of the data ingestion pipeline
        obj.main()
        # Log the successful completion of the data ingestion stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        raise e  # Reraise the exception for further handling if necessary


# Define the name of the current stage in the data processing pipeline
STAGE_NAME = "Prepare Base Model stage"

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
