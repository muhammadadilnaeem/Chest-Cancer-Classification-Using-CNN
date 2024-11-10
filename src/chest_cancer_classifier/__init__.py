
import os  # Import the os module for interacting with the operating system
import sys  # Import the sys module for system-specific parameters and functions
import logging  # Import the logging module for logging messages

# Define the logging format string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Specify the directory where log files will be stored
log_dir = "logs"

# Create the full path for the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it does not exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Use the defined format for log messages
    handlers=[
        logging.FileHandler(log_filepath),  # Log messages to a file
        logging.StreamHandler(sys.stdout)    # Also output log messages to the console
    ]
)

# Create a logger object with a specific name
logger = logging.getLogger("chest_cancer_classifier_logger")  # This logger can be used throughout the application