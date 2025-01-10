
# import libraries

import os
import time
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
import urllib.request as request
from chest_cancer_classifier.entity.config_entity import TrainingConfig



class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initialize the Training class with a TrainingConfig object.

        :param config: TrainingConfig object containing configuration for training.
        """
        self.config = config

    def get_base_model(self):
        """
        Load and compile the base model for training.
        """
        # Load the pre-trained base model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Compile the model with optimizer, loss, and metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),  # Adam optimizer for training
            loss="binary_crossentropy",  # Loss for binary classification
            metrics = ["accuracy"]  # Metrics to track model performance
        )

    def train_valid_generator(self):
        """
        Prepare training, validation, and test data generators.
        """
        # Arguments for data preprocessing
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
            validation_split=0.20
        )

        # Arguments for image resizing and batching
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Image size excluding channels
            batch_size=self.config.params_batch_size,  # Batch size for generators
            interpolation="bilinear"  # Interpolation method for resizing images
        )

        # Validation data generator using 'valid' folder
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,  # No shuffling for validation data
            **dataflow_kwargs
        )

        # Training data generator
        if self.config.params_is_augmentation:
            # Augmentation for training data if enabled in config
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Random rotation
                horizontal_flip=True,  # Random horizontal flips
                width_shift_range=0.2,  # Horizontal shift
                height_shift_range=0.2,  # Vertical shift
                shear_range=0.2,  # Shear transformation
                zoom_range=0.2,  # Zoom transformation
                **datagenerator_kwargs
            )
        else:
            # Simple generator without augmentation
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,  # Shuffle training data
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.

        :param path: Path to save the model.
        :param model: Trained model object.
        """
        model.save(path)

    def train(self):
        """
        Train the model using the training and validation data generators.
        """
        # Calculate steps per epoch for training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,  # Training data generator
            epochs=self.config.params_epochs,  # Number of epochs
            steps_per_epoch=self.steps_per_epoch,  # Steps per epoch
            validation_data=self.valid_generator,  # Validation data generator
            validation_steps=self.validation_steps  # Validation steps
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,  # Save path for trained model
            model=self.model  # The trained model
        )






## For binary classification

# class Training:
#     def __init__(self, config: TrainingConfig):
#         # Initialize the Training class with a TrainingConfig object
#         self.config = config

#     def get_base_model(self):
#         # Load the base model from the specified path in the configuration
#         self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

#         # Compile the model with an optimizer and loss function suitable for the task
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(),  # Using Adam optimizer for training
#             loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
#             metrics = ["accuracy", "precision", "recall", "f1_score"]  # Track accuracy as a performance metric
#         )

#     def train_valid_generator(self):
#         # Set up data generator arguments for preprocessing the images
#         datagenerator_kwargs = dict(
#             rescale=1.0 / 255,  # Normalize the pixel values to the range [0, 1]
#             validation_split=0.20  # Use 20% of the data for validation
#         )

#         # Set up data flow arguments for resizing images and defining batch size
#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],  # Resize images to specified dimensions (excluding channels)
#             batch_size=self.config.params_batch_size,  # Set the batch size for training and validation
#             interpolation="bilinear"  # Set the interpolation method for resizing images
#         )

#         # Create a validation data generator
#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )
#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,  # Directory containing training and validation data
#             subset="validation",  # Specify that this generator is for validation data
#             shuffle=False,  # Do not shuffle validation data
#             class_mode="sparse",  # Use "sparse" mode for integer labels
#             **dataflow_kwargs  # Include data flow arguments for resizing and batch size
#         )

#         # Create a training data generator with or without augmentation
#         if self.config.params_is_augmentation:
#             # If augmentation is enabled, configure the data generator with augmentation techniques
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,  # Randomly rotate images within a range
#                 horizontal_flip=True,  # Randomly flip images horizontally
#                 width_shift_range=0.2,  # Randomly shift images horizontally
#                 height_shift_range=0.2,  # Randomly shift images vertically
#                 shear_range=0.2,  # Apply random shearing transformations
#                 zoom_range=0.2,  # Randomly zoom into images
#                 **datagenerator_kwargs  # Include normalization and validation split
#             )
#         else:
#             # If no augmentation is needed, use the validation data generator
#             train_datagenerator = valid_datagenerator

#         # Generate training data from the directory
#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,  # Directory containing training data
#             subset="training",  # Specify that this generator is for training data
#             shuffle=True,  # Shuffle training data
#             class_mode="sparse",  # Use "sparse" mode for integer labels
#             **dataflow_kwargs  # Include data flow arguments for resizing and batch size
#         )

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         # Static method to save the trained model to the specified path
#         model.save(path)

#     def train(self):
#         # Calculate the number of steps per epoch based on training data
#         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
#         # Calculate the number of validation steps based on validation data
#         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

#         # Start the training process
#         self.model.fit(
#             self.train_generator,  # Training data generator
#             epochs=self.config.params_epochs,  # Number of epochs specified in the configuration
#             steps_per_epoch=self.steps_per_epoch,  # Steps per epoch
#             validation_steps=self.validation_steps,  # Steps for validation
#             validation_data=self.valid_generator  # Validation data generator
#         )

#         # Save the trained model to the specified path
#         self.save_model(
#             path=self.config.trained_model_path,  # Path where the trained model will be saved
#             model=self.model  # The trained model to save
#         )