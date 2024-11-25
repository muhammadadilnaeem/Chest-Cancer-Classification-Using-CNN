
# complete pipeline 

from src.chest_cancer_classifier import logger
from chest_cancer_classifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from chest_cancer_classifier.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline
from chest_cancer_classifier.pipeline.stage_3_model_training import ModelTrainingPipeline
from chest_cancer_classifier.pipeline.stage_4_model_evaluation import EvaluationPipeline

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
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        
        # Log the completion of the stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        raise e  # Reraise the exception for further handling


# Define the name of the current stage in the data processing pipeline
STAGE_NAME = "Model Training stage"

try:
        # Log the start of the stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the training pipeline and run the main method
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        
        # Log the completion of the stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        raise e  # Reraise the exception for further handling


# Define the name of the current stage in the data processing pipeline
STAGE_NAME = "Model Evaluation stage"

try:
        # Log the start of the evaluation stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the EvaluationPipeline class and run the main method
        model_evaluater = EvaluationPipeline()
        model_evaluater.main()
        
        # Log the completion of the evaluation stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        # Raise the exception to propagate it further
        raise e