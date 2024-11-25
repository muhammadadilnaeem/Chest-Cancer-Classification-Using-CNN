
-----

# **`Chest Cancer Classification Using MLflow and DVC`**

This repository demonstrates the implementation of a **chest cancer classification** pipeline using machine learning techniques for **multi class calssification**. It integrates **MLflow** for experiment tracking and **DVC (Data Version Control)** for managing datasets and models. The project showcases an end-to-end machine learning pipeline, including CI/CD deployment to AWS using GitHub Actions.


https://github.com/user-attachments/assets/96fc046d-6598-445e-baaf-4ade6296000c


> **Note**: This is a practice project and is not intended for clinical or production use.

---

## Table of Contents

- [**`Chest Cancer Classification Using MLflow and DVC`**](#chest-cancer-classification-using-mlflow-and-dvc)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Project Workflows](#project-workflows)
    - [Development Workflow](#development-workflow)
    - [AWS CI/CD Deployment Workflow](#aws-cicd-deployment-workflow)
  - [Setup Instructions](#setup-instructions)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
    - [Set Up DVC](#set-up-dvc)
    - [Set Up MLflow](#set-up-mlflow)
    - [Configure AWS Credentials](#configure-aws-credentials)
  - [How to Run](#how-to-run)
    - [Preprocess the Data](#preprocess-the-data)
    - [Train and Log Experiments](#train-and-log-experiments)
    - [Evaluate the Model](#evaluate-the-model)
    - [Push Changes with DVC](#push-changes-with-dvc)
    - [Trigger CI/CD Pipeline](#trigger-cicd-pipeline)
  - [Repository Structure](#repository-structure)
  - [Future Enhancements](#future-enhancements)
  - [Acknowledgments](#acknowledgments)
  - [Disclaimer](#disclaimer)

---

## Introduction

Cancer classification is a crucial application of machine learning in healthcare. This project focuses on developing a classification model for chest cancer. It includes:

- **Data pre-processing**  
- **Feature extraction and engineering**  
- **Model training and evaluation**  
- **Experiment tracking with MLflow**  
- **Versioning datasets and models with DVC**  
- **CI/CD pipeline for deployment to AWS**  

The goal of this project is to showcase reproducible machine learning pipelines and end-to-end deployment workflows.

---

## Features

- **End-to-End Machine Learning Workflow**: Covers data processing, training, evaluation, and deployment.  
- **Experiment Tracking**: Using MLflow to log and compare experiments.  
- **Data and Model Versioning**: Managed with DVC for reproducibility.  
- **CI/CD Deployment**: Automated deployment to AWS using GitHub Actions.  
- **Scalable Infrastructure**: Demonstrates cloud deployment principles.

---

## Technologies Used

- **Python**: Primary language for data science and machine learning.  
- **MLflow**: Tracks experiments and manages model lifecycle.  
- **DVC**: Handles data and model versioning.  
- **Scikit-learn**: Implements machine learning models.  
- **AWS (EC2, S3)**: Hosts the deployed application.  
- **GitHub Actions**: Automates CI/CD workflows.  
- **Docker**: Creates containers for deployment.  
- **Pandas & NumPy**: For data manipulation and analysis.  
- **Matplotlib & Seaborn**: For visualizations.  

---

## Project Workflows

### Development Workflow

1. **Data Collection and Management**  
   - Manage datasets with **DVC**, ensuring version control.  
   - Store raw data files in the `data/` directory.  

2. **Data Preprocessing**  
   - Clean the dataset (handle missing values, outliers, etc.).  
   - Normalize or scale features for model compatibility.  

3. **Feature Engineering**  
   - Extract relevant features to enhance model performance.  

4. **Model Training and Experiment Tracking**  
   - Train multiple models and track performance with MLflow.  
   - Log hyperparameters, metrics, and artifacts for comparison.

5. **Model Evaluation**  
   - Evaluate models on test data using metrics like accuracy, precision, recall, and F1 score.  
   - Select the best-performing model.  

6. **Versioning with DVC**  
   - Version datasets and trained models using DVC.  
   - Push data to remote storage to ensure reproducibility.

---

### AWS CI/CD Deployment Workflow

1. **Build Docker Image**  
   - Create a `Dockerfile` to define the application environment.  

2. **Set Up AWS Infrastructure**  
   - Launch an EC2 instance for hosting the application.  
   - Set up S3 buckets for storing datasets and artifacts.  
   - Configure IAM roles and security groups.  

3. **Write GitHub Actions Workflow**  
   - Define CI/CD pipeline in `.github/workflows/deploy.yml`.  
   - Steps include:  
     - Building and testing the Docker image.  
     - Pushing the Docker image to a container registry (e.g., Amazon ECR or Docker Hub).  
     - Deploying the application to the EC2 instance.

4. **Deployment Pipeline Steps**  
   - Trigger the pipeline on `push` or `pull_request`.  
   - Build and package the application.  
   - Deploy the updated version to AWS.

---

## Setup Instructions

Follow these steps to set up the project on your local machine:

### Clone the Repository

```bash
git clone https://github.com/muhammadadilnaeem/Chest-Cancer-Classification-Using-MLflow-and-DVC.git
cd Chest-Cancer-Classification-Using-MLflow-and-DVC
```

### Install Dependencies

Set up a Python virtual environment and install the required libraries:  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Set Up DVC

Initialize DVC and pull data from remote storage:  
```bash
dvc init
dvc pull
```

### Set Up MLflow

Ensure MLflow is installed and run the tracking server:  
```bash
mlflow ui
```

### Configure AWS Credentials

- Set up AWS CLI and configure credentials:  
  ```bash
  aws configure
  ```
- Ensure your EC2 instance has the necessary IAM roles and permissions.

---

## How to Run

### Preprocess the Data

```bash
python scripts/preprocess_data.py
```

### Train and Log Experiments

```bash
python scripts/train_model.py
```

### Evaluate the Model

```bash
python scripts/evaluate_model.py
```

### Push Changes with DVC

Version datasets and models:  
```bash
dvc add data/processed_data.csv
dvc push
```

### Trigger CI/CD Pipeline

- Commit and push changes to GitHub:  
  ```bash
  git add .
  git commit -m "Updated pipeline"
  git push origin main
  ```
- GitHub Actions will automatically build, test, and deploy the application.

---

## Repository Structure

```plaintext
Chest-Cancer-Classification-Using-MLflow-and-DVC/
│
├── .github/workflows/           # GitHub Actions workflows
│   └── deploy.yml               # CI/CD pipeline definition
├── data/                        # Raw and processed datasets
├── models/                      # Trained model files
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # Python scripts for various stages
│   ├── preprocess_data.py       # Preprocessing script
│   ├── train_model.py           # Training script
│   ├── evaluate_model.py        # Evaluation script
│
├── Dockerfile                   # Docker image configuration
├── dvc.yaml                     # DVC pipeline file
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Ignored files
└── .dvcignore                   # Ignored files for DVC
```

---

## Future Enhancements

- Extend the pipeline to include deep learning models (e.g., CNNs).  
- Integrate deployment with **Kubernetes** for better scalability.  
- Add automated testing for the ML pipeline using **pytest**.  
- Create a detailed dashboard for visualizing model performance.  

---

## Acknowledgments

This project is inspired by real-world challenges in medical image analysis. It serves as a practical example for demonstrating best practices in machine learning workflows and cloud deployments.

---

## Disclaimer

This project is created **solely for practice and learning purposes**. It is not intended for clinical or production use. The models and results are not validated for real-world medical applications. This project is inspired from [this youtube vedio](https://youtu.be/-NOIWzjJK-4?si=37m-vc_ESKi1pjV0) .

---
