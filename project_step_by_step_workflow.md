

------

# **`Step By Step Project Workflow`**

1. **Create** a Github **Repository**.
    - Add **licence**, **README.md** and **.gitignore** file.
2. **Clone** the Repository in **Your system**.

3. It is a good approach to create a **Project Template**. It help to **reduce effort** to **create files and folders manually**.
    - Create a **template.py file**. Add code in that file. Here is the code for this project's **template.py file**.

    ```bash
    # Import necessory libraries for setting up project template

    import os
    import logging
    from pathlib import Path

    # Set up information level log for tracking progress
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

    # Set up Project name
    project_name = "chest_cancer_classifier"

    # Files that i need for this project
    list_of_files = [
        ".github/workflows/.gitkeep",
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/components/__init__.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/config/__init__.py",
        f"src/{project_name}/config/configuration.py",
        f"src/{project_name}/pipeline/__init__.py",
        f"src/{project_name}/entity/__init__.py",
        f"src/{project_name}/constants/__init__.py",
        "config/config.yaml",
        "dvc.yaml",
        "params.yaml",
        "requirements.txt",
        "setup.py",
        "research/experiments.ipynb",
        "templates/index.html",
        "static/style.css",
        "app.py"
    ]


    # Iterate over each file path in the list of files
    for filepath in list_of_files:
        # Convert the filepath string to a Path object for easier manipulation
        filepath = Path(filepath)
        
        # Split the filepath into its directory and filename components
        filedir, filename = os.path.split(filepath)

        # Check if the directory part of the filepath is not empty
        if filedir != "":
            # Create the directory if it doesn't exist, without raising an error if it does
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory; {filedir} for the file: {filename}")

        # Check if the file does not exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            # Open the file in write mode (this creates the file if it doesn't exist)
            with open(filepath, "w") as f:
                pass  # No content is written to the file
                logging.info(f"Creating empty file: {filepath}")

        else:
            # Log that the file already exists
            logging.info(f"{filename} already exists.")
    ``` 

    - Now after writing your code it's time to implement this.
        - Go to your VS Code **terminal** and run this file with this command.
            ```bash
            python template.py
            ```
        - You will see the whole buch of **files and folders** will be created **automatically**.    

4. Next step will be **setting up Project** and **Virtual Environment**.
   
    - Let's First create a **Virtual Environment**. In my case I am using **conda**. 
      - First command will be to **create** a virtual environment with **specific python version** and **-y flag mean yes to all comming steps**.

          ```bash
          conda create -p venv python=3.10 -y 
          ```
      - Now we need to activate the **Created Virtual Environment** with this command.        
          ```bash
          conda activate venv 
          ```
      - Now we need to **install specific libraries in this environnment** according to our specific **use case**. **You can install it 1 by 1**.
      - But more convinent approach wil be to create a **requirements.txt**. Add names of libraries in that file. At the end add `-e .` for project setup.
      - Next you need to create a **setup.py** file for project setup. This will **contain information related to your project** (author name, project name, libraries used in this project). Here is a the code of this project's **setup.py**.

        ```bash
        # Import the setuptools library to facilitate packaging and distribution
        import setuptools

        # Open the README file to read its content for the long description, using UTF-8 encoding
        with open("README.md", "r", encoding="utf-8") as f:
            long_description = f.read()  # Store the README content

        # Open the requirements.txt file to read all dependencies
        with open("requirements.txt") as f:
            install_requires = f.read().splitlines()  # Split lines to get each dependency as a list item

        # Define the version of the package
        __version__ = "0.0.0"

        # Define metadata for the package
        REPO_NAME = "Chest-Cancer-Classification-Using-MLflow-and-DVC"  # GitHub repository name
        AUTHOR_USER_NAME = "muhammadadilnaeem"  # GitHub username
        SRC_REPO = "chest_cancer_classifier"  # Name of the source directory/package
        AUTHOR_EMAIL = "madilnaeem0@gmail.com"  # Contact email for the author

        # Call the setup function to configure the package
        setuptools.setup(
            name=SRC_REPO,  # Name of the package, must match the directory in 'src'
            version=__version__,  # Version of the package
            author=AUTHOR_USER_NAME,  # Author name as it will appear in package metadata
            author_email=AUTHOR_EMAIL,  # Author's contact email
            description="A demo python package for Chest Cancer Classification Web Application.",  # Short package description
            long_description=long_description,  # Detailed description read from README.md
            long_description_content_type="text/markdown",  # Format of the long description (markdown)
            url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the package repository
            project_urls={  # Additional URLs related to the project, such as an issue tracker
                "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
            },
            package_dir={"": "src"},  # Specify that the root of packages is the 'src' directory
            packages=setuptools.find_packages(where="src"),  # Automatically find all packages in 'src'
            install_requires=install_requires,  # List of dependencies read from requirements.txt
            classifiers=[  # Additional metadata classifiers that help others find the package
                "Programming Language :: Python :: 3",  # Indicate that the package is written in Python 3
                "License :: OSI Approved :: MIT License",  # Specify the license type
                "Operating System :: OS Independent",  # OS compatibility
            ],
            python_requires='>=3.10',  # Specify the minimum Python version required
        )      
        ```  

      -  But i will install libraries using **requirements.txt** with this command. This will install all mentioned libraries in **requirements.txt** and set up project.
        
            ```bash
            pip install -r requirements.txt 
            ```

      - This is an `optional step`. If you want to check what **specific versions of libraries are installed in Your Virtual Environment** use this command.
        ```bash
        pip freeze > requirements.txt
        ``` 
        

5. Next step would be to set up **logging and exception** for better code readibilty and pracrice.
    - We need to **set up loging**.
        - Inside `src` folder we have `__init__.py`. We will write logging code in this file.It will help us direcly import `logger`. Here is what logging code will look like
            ```bash
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
            ```  

    - We need to **set up exceptions**. For this project We will set this up using a python package `python-box`. All exceptions code can be found in `common_functions.py` which we will discuss in next step. But first we will use some new packages for this prject and we need to take a look:

        ### **python-box**

        - **Explanation**:  
        `python-box` is a versatile library that extends Python dictionaries, allowing you to access dictionary keys as object attributes (dot-notation). It also includes advanced functionality, such as working seamlessly with JSON, YAML, and other data structures.

        - **Use Case**:  
        - Ideal for managing configuration data or JSON-like structures in an easy-to-access and user-friendly manner.

        - **Example**:  
        ```python
        from box import Box

        # Original dictionary
        dic = {"name": "John", "age": 25}

        # Convert to a Box object
        dic = Box(dic)

        # Access values using dot notation
        print(dic.name)  # Output: John
        print(dic.age)   # Output: 25

        # Adding new key-value pairs
        dic.country = "USA"
        print(dic.country)  # Output: USA
        ```

        - **Additional Features**:
        - Handles nested dictionaries for easy access.
        - Converts JSON/YAML files directly into Box objects for seamless integration with configuration files.


        ### **config-box**
        - **Explanation**: 
        - `config-box` provides a way to handle Python dictionaries as objects, enabling dot-notation access (e.g., `obj.key`) instead of traditional dictionary indexing (`obj["key"]`). It is especially helpful when working with configuration files like YAML or JSON.

        - **Use Case**: 
        - Simplifies working with nested dictionary configurations.

        - **Example**:
        ```python
        from box import ConfigBox

        # Original dictionary
        dic = {"key": "value", "tala": "chabi"}

        # Convert to ConfigBox
        dic = ConfigBox(dic)

        # Access value with dot notation
        print(dic.tala)  # Output: chabi
        ```

        ### **ensure**
        - **Explanation**: 
        - `ensure` enforces type annotations at runtime. If the provided arguments don't match the annotated types, it raises an error. This is particularly useful for maintaining strict type-checking in functions.

        - **Use Case**: 
        - Ensures functions only receive arguments of expected types.

        - **Example**:
        ```python
        from ensure import ensure_annotations

        @ensure_annotations
        def get_multiplication(x: int, y: int):
            return x * y

        # Correct usage
        print(get_multiplication(x=6, y=9))  # Output: 54

        # Incorrect usage
        print(get_multiplication(x=6, y="9"))  # Raises TypeError: argument 'y' must be int
        ``` 





6. Now it's time to create **functions** that w**e will use again and again in our project**. Instead of writing code again and again for different functions we will just write code and keep it in `common_functions.py`, whenever we need any functionality we will just import that code from this file. 
   - For this I will go to `src` and then `utils` folder.
   - Inside this folder we will create a file `common_functions.py`, which will look like this
        ```bash
        # Import libraries
        import os
        import json
        import yaml
        import joblib
        import base64
        from typing import Any
        from pathlib import Path
        from box import ConfigBox
        from ensure import ensure_annotations
        from box.exceptions import BoxValueError
        from chest_cancer_classifier import logger

        @ensure_annotations
        def read_yaml(path_to_yaml: Path) -> ConfigBox:
            """
            Reads a YAML file and returns its contents as a ConfigBox object.

            Args:
                path_to_yaml (Path): Path to the YAML file.

            Raises:
                ValueError: If the YAML file is empty.
                Exception: If any other error occurs during file reading.

            Returns:
                ConfigBox: Parsed content from YAML as a ConfigBox object.
            """
            try:
                with open(path_to_yaml) as yaml_file:
                    content = yaml.safe_load(yaml_file)
                    logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
                    return ConfigBox(content)
            except BoxValueError:
                logger.error(f"YAML file '{path_to_yaml}' is empty")
                raise ValueError("YAML file is empty")
            except Exception as e:
                logger.exception(f"Error loading YAML file '{path_to_yaml}': {e}")
                raise e


        @ensure_annotations
        def create_directories(path_to_directories: list, verbose=True):
            """
            Creates a list of directories if they do not already exist.

            Args:
                path_to_directories (list): List of directory paths to create.
                verbose (bool, optional): If True, logs creation for each directory. Defaults to True.
            """
            for path in path_to_directories:
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logger.info(f"Directory created at: {path}")


        @ensure_annotations
        def save_json(path: Path, data: dict):
            """
            Saves a dictionary as a JSON file.

            Args:
                path (Path): Path where JSON file will be saved.
                data (dict): Data to save in JSON format.
            """
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"JSON file saved at: {path}")


        @ensure_annotations
        def load_json(path: Path) -> ConfigBox:
            """
            Loads JSON data from a file and returns it as a ConfigBox object.

            Args:
                path (Path): Path to the JSON file.

            Returns:
                ConfigBox: Data loaded from JSON file as a ConfigBox.
            """
            with open(path) as f:
                content = json.load(f)
            logger.info(f"JSON file loaded successfully from: {path}")
            return ConfigBox(content)


        @ensure_annotations
        def save_bin(data: Any, path: Path):
            """
            Saves data in binary format using joblib.

            Args:
                data (Any): Data to save as binary.
                path (Path): Path where binary file will be saved.
            """
            joblib.dump(value=data, filename=path)
            logger.info(f"Binary file saved at: {path}")


        @ensure_annotations
        def load_bin(path: Path) -> Any:
            """
            Loads binary data using joblib.

            Args:
                path (Path): Path to the binary file.

            Returns:
                Any: Object stored in the binary file.
            """
            data = joblib.load(path)
            logger.info(f"Binary file loaded from: {path}")
            return data


        @ensure_annotations
        def get_size(path: Path) -> str:
            """
            Gets the file size in kilobytes (KB).

            Args:
                path (Path): Path of the file.

            Returns:
                str: Size in KB, rounded to the nearest integer.
            """
            size_in_kb = round(os.path.getsize(path) / 1024)
            logger.info(f"Size of '{path}': ~{size_in_kb} KB")
            return f"~ {size_in_kb} KB"


        def decodeImage(imgstring: str, fileName: str):
            """
            Decodes a base64 string and saves it as an image file.

            Args:
                imgstring (str): Base64-encoded image string.
                fileName (str): Path where the decoded image will be saved.
            """
            imgdata = base64.b64decode(imgstring)
            with open(fileName, 'wb') as f:
                f.write(imgdata)
            logger.info(f"Image decoded and saved to '{fileName}'")


        def encodeImageIntoBase64(croppedImagePath: str) -> str:
            """
            Encodes an image file as a base64 string.

            Args:
                croppedImagePath (str): Path to the image file.

            Returns:
                str: Base64-encoded string of the image file.
            """
            with open(croppedImagePath, "rb") as f:
                encoded_string = base64.b64encode(f.read())
            logger.info(f"Image at '{croppedImagePath}' encoded to base64")
            return encoded_string.decode('utf-8')
        ```

7. Now we need to understand **data Acquisition** for this. You can get data from any where like from databases (mongodb,sql), platforms like (kaggle, uci machine learning).
    - But for this project we will acquire data from `Google Drive`.

8. Now we have **set up project**, We need to **understand how to write our code step by step**.  

    ### **Project Workflow**

    0. **Perform Experiments in `research` folder**:
   
    - For every stage we will create seperate **Jupyter notebook**, we will write code in **Jupyter notebook (.ipynb)** for **experiments** and use same code in **python files (.py)**. Here are the names of these files:
   
      - `01_data_ingestion.ipynb`         
      - `02_prepare_base_model.ipynb`         
      - `03_model_trainer.ipynb`         
      - `04_model_evaluation.ipynb`         

    1. **Update `config.yaml`**:  
    Define project-wide static configurations, such as file paths, thresholds, or model parameters.

    2. **Update `secrets.yaml` (Optional)**:  
    Store sensitive data like API keys or passwords securely (if required).

    3. **Update `params.yaml`**:  
    Define tunable parameters for experiments, such as learning rates or batch sizes.
       - We will load parameters of `VGG16` 

    4. **Update the Entity**:  
    Create structured classes to represent data models, ensuring type safety and clarity.
        - In our case we have file `config_entity.py`. This file has all entities used in this project for configuration.

    5. **Update the Configuration Manager**:  
    Implement a utility in `src/config` to parse and manage configuration files dynamically.
        - In our case we have `configuration.py`.

    6. **Update the Components**:  
    Write modular functions or classes in `src/components` to perform specific tasks like data preprocessing or model training. Here are the folloowing components used in this project:
        - `data_ingestion.py`.
        - `prepare_base_models.py`.
        - `model_trainer.py`.
        - `prepare_base_models.py`.

    7. **Update the Pipeline**:  
    Integrate components in `src/pipeline` to define end-to-end workflows (e.g., data processing to model inference).
     - Our pipeline stages step by step will be 
      
        - `stage_1_data_ingestion.py`
        - `stage_2_prepare_base_model.py`
        - `stage_3_model_training.py`
        - `stage_4_model_evaluation.py`
        - `stage_5_prediction.py`

    8.  **Update the `main.py`**:  
    Write the entry point script to execute the pipeline and orchestrate the workflow.

    9.  **Update the `dvc.yaml`**:  
    Define and version your data pipeline with DVC, including stages for data preparation, model training, and evaluation.

9. We can **track experiments** that we perform with our code using **Open Source** `MLops Tools` like **Dagshub**, **MLflow** and **DVC**.

    - For this project we will first set up our Dagshub account and then use our **github repository** in **Dagshub** for **experiment tracking using MLflow**.

        - An example of MLflow code is:
      
            ```bash
            def log_into_mlflow(self):
                    # Set the MLflow tracking URI from the configuration
                    mlflow.set_registry_uri(self.config.mlflow_uri)
                    
                    # Parse the tracking URI to determine the type of storage used
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    
                    # Set the experiment name
                    experiment_name = "Model Evaluation Experimentation" 
                    mlflow.set_experiment(experiment_name)

                    # Start a new MLflow run to log parameters and metrics
                    with mlflow.start_run(run_name="Model Evaluation"):
                        # Log parameters from the configuration
                        mlflow.log_params(self.config.all_params)
                        # Log evaluation metrics (loss and accuracy)
                        mlflow.log_metrics(
                            {"loss": self.score[0], "accuracy": self.score[1]}
                        )
                        # Check if the tracking URL is not a file store
                        if tracking_url_type_store != "file":
                            # Register the model with MLflow
                            mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                                            
                        else:
                            # Log the model without registration if using file store
                            mlflow.keras.log_model(self.model, "model")
            ```

        - Now to integrate **DVC** for **Pipeline Tracking** () use these commands:

            - First step is to initialize DVC:
                
                ```bash
                dvc init
                ```

            - Second step will be to run this command for pipeline tracking which will run dvc.yaml to perform this:
             
                ```bash
                dvc repro
                ```

            - Now to see graph of how the whole pipeline connected (dependecy of the pipeline) run this command:

                ```bash
                dvc dag
                ```            

