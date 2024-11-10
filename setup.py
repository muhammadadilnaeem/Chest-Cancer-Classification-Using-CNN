
# Import the setuptools package to facilitate packaging and distribution
import setuptools 

# Open the README file to read its content, using UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()  # Store the content of the README for the long description

# Define the version of the package
__version__ = "0.0.0"

# Define metadata for the package
REPO_NAME = "Chest-Cancer-Classification-Using-MLflow-and-DVC"  # Name of the repository
AUTHOR_USER_NAME = "muhammadadilnaeem"  # Author's GitHub username
SRC_REPO = "chest_cancer_classifier"  # Name of the source repository/package
AUTHOR_EMAIL = "madilnaeem0@gmail.com"  # Author's contact email

# Call the setup function to configure the package
setuptools.setup(
    name=SRC_REPO,  # Name of the package
    version=__version__,  # Version of the package
    author=AUTHOR_USER_NAME,  # Author of the package
    author_email=AUTHOR_EMAIL,  # Author's email address
    description="A demo python package for Chest Cancer Classification Web Application.",  # Short description of the package
    long_description=long_description,  # Detailed description read from the README file
    long_description_content_type="text/markdown",  # Specify the format of the long description
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL for the package repository
    project_urls={  # Additional URLs related to the project
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",  # Link to the issue tracker
    },
    package_dir={"": "src"},  # Specify that packages are located in the 'src' directory
    packages=setuptools.find_packages(where="src")  # Automatically find all packages in the 'src' directory
)