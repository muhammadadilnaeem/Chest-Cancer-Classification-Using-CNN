
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
        # Initialize the Training class with a TrainingConfig object
        self.config = config

    def get_base_model(self):
        # Load the base model from the specified path in the configuration
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Compile the model with an optimizer and loss function suitable for the task
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),  # Using Adam optimizer for training
            loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
            metrics=['accuracy']  # Track accuracy as a performance metric
        )

    def train_valid_generator(self):
        # Set up data generator arguments for preprocessing the images
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Normalize the pixel values to the range [0, 1]
            validation_split=0.20  # Use 20% of the data for validation
        )

        # Set up data flow arguments for resizing images and defining batch size
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize images to specified dimensions (excluding channels)
            batch_size=self.config.params_batch_size,  # Set the batch size for training and validation
            interpolation="bilinear"  # Set the interpolation method for resizing images
        )

        # Create a validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory containing training and validation data
            subset="validation",  # Specify that this generator is for validation data
            shuffle=False,  # Do not shuffle validation data
            class_mode="sparse",  # Use "sparse" mode for integer labels
            **dataflow_kwargs  # Include data flow arguments for resizing and batch size
        )

        # Create a training data generator with or without augmentation
        if self.config.params_is_augmentation:
            # If augmentation is enabled, configure the data generator with augmentation techniques
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Randomly rotate images within a range
                horizontal_flip=True,  # Randomly flip images horizontally
                width_shift_range=0.2,  # Randomly shift images horizontally
                height_shift_range=0.2,  # Randomly shift images vertically
                shear_range=0.2,  # Apply random shearing transformations
                zoom_range=0.2,  # Randomly zoom into images
                **datagenerator_kwargs  # Include normalization and validation split
            )
        else:
            # If no augmentation is needed, use the validation data generator
            train_datagenerator = valid_datagenerator

        # Generate training data from the directory
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory containing training data
            subset="training",  # Specify that this generator is for training data
            shuffle=True,  # Shuffle training data
            class_mode="sparse",  # Use "sparse" mode for integer labels
            **dataflow_kwargs  # Include data flow arguments for resizing and batch size
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Static method to save the trained model to the specified path
        model.save(path)

    def train(self):
        # Calculate the number of steps per epoch based on training data
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # Calculate the number of validation steps based on validation data
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Start the training process
        self.model.fit(
            self.train_generator,  # Training data generator
            epochs=self.config.params_epochs,  # Number of epochs specified in the configuration
            steps_per_epoch=self.steps_per_epoch,  # Steps per epoch
            validation_steps=self.validation_steps,  # Steps for validation
            validation_data=self.valid_generator  # Validation data generator
        )

        # Save the trained model to the specified path
        self.save_model(
            path=self.config.trained_model_path,  # Path where the trained model will be saved
            model=self.model  # The trained model to save
        )