
from chest_cancer_classifier.config.configuration import ConfigurationManager
from chest_cancer_classifier.components.model_evaluation_with_mlflow import Evaluation
from chest_cancer_classifier import logger


# Define the name of the current stage
STAGE_NAME = "Evaluation stage"

# Class to handle the evaluation pipeline
class EvaluationPipeline:
    def __init__(self):
        # Initialize any necessary attributes here (currently none)
        pass

    def main(self):
        # Load the configuration manager to get evaluation settings
        config = ConfigurationManager()
        # Retrieve evaluation configuration
        eval_config = config.get_evaluation_config()
        
        # Create an instance of the Evaluation class with the retrieved configuration
        evaluation = Evaluation(eval_config)
        # Perform the evaluation
        evaluation.evaluation()
        # Save the evaluation score
        evaluation.save_score()
        # Optionally log the evaluation results into MLflow (commented out)
        evaluation.log_into_mlflow()

# Entry point of the script
if __name__ == '__main__':
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