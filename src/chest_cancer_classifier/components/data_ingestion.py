import os  # Import the os module for operating system functionalities
import zipfile  # Import the zipfile module to handle ZIP file extraction
import gdown  # Import gdown for downloading files from Google Drive
from chest_cancer_classifier import logger  # Import the logger for logging events
from chest_cancer_classifier.utils.common_functions import get_size  # Import utility function to get file size
from chest_cancer_classifier.entity.config_entity import DataIngestionConfig  # Import the DataIngestionConfig class

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Class to handle data ingestion processes, including downloading and extracting data
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config  # Store the configuration object passed during initialization

    def download_file(self) -> str:
        '''
        Fetch data from the URL specified in the configuration.
        Returns the path to the downloaded file.
        '''
        try:
            dataset_url = self.config.source_URL  # Get the data source URL from the config
            zip_download_dir = self.config.local_data_file  # Path where the downloaded file will be saved
            
            # Create the directory for data ingestion if it doesn't exist
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")  # Log the download start

            # Extract the file ID from the Google Drive URL
            file_id = dataset_url.split("/")[-2]
            # Construct the download link for gdown
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Set up a requests session with retries and timeout
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("https://", adapter)
            
            with session:
                # Use gdown with session to download the file
                gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")  # Log successful download

        except requests.exceptions.ConnectTimeout:
            logger.error("Connection timed out. Please check your internet connection or try again later.")
        except Exception as e:
            logger.error(f"An error occurred during download: {e}")
            raise e  # Raise any other exceptions encountered during the download    

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory specified in the configuration.
        Function returns None.
        """
        unzip_path = self.config.unzip_dir  # Get the directory where the zip file will be extracted
        os.makedirs(unzip_path, exist_ok=True)  # Create the extraction directory if it doesn't exist
        
        # Open the zip file and extract its contents
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)  # Extract all files to the specified directory