

from chest_cancer_classifier import logger
from chest_cancer_classifier.components.data_ingestion import DataIngestion
from chest_cancer_classifier.config.configuration import DataIngestionConfig, ConfigurationManager

# Define the name of the current stage in the data processing pipeline
STAGE_NAME = "Data Ingestion stage"

# Class to manage the data ingestion training pipeline
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass  # Constructor does not require any initialization parameters

    def main(self):
        # Create an instance of the ConfigurationManager to load configurations
        config = ConfigurationManager()
        # Retrieve the data ingestion configuration
        data_ingestion_config = config.get_data_ingestion_config()
        # Create an instance of DataIngestion with the loaded configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # Download the data file from the configured URL
        data_ingestion.download_file()
        # Extract the contents of the downloaded ZIP file
        data_ingestion.extract_zip_file()

# Entry point for the script
if __name__ == '__main__':
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