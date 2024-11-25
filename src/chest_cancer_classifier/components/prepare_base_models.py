
import os
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
import urllib.request as request
from chest_cancer_classifier.entity.config_entity import (DataIngestionConfig,
                                                              PrepareBaseModelConfig,
                                                              TrainingConfig,
                                                              EvaluationConfig)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        # Initialize the class with a configuration object for preparing the base model
        self.config = config

    
    def get_base_model(self):
        # Create the base model using the VGG16 architecture from Keras
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,  # Set the input shape based on the config
            weights=self.config.params_weights,  # Load weights specified in the config
            include_top=self.config.params_include_top  # Include the top layer or not, based on config
        )

        # Save the created base model to the specified path
        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Prepare a full model by adding a classification layer on top of the base model
        
        # Freeze all layers if freeze_all is True
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        # Freeze layers until the specified layer index if freeze_till is provided
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Flatten the output from the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        # Add a dense layer for predictions with softmax activation
        prediction = tf.keras.layers.Dense(
            units=classes,  # Number of classes for the output
            activation="softmax"  # Softmax activation for multi-class classification
        )(flatten_in)

        # Create the full model with the specified inputs and outputs
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the full model with the specified optimizer and loss function
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),  # Stochastic Gradient Descent optimizer
            loss=tf.keras.losses.CategoricalCrossentropy(),  # Loss function for multi-class classification
            metrics = ["accuracy"]  # Track accuracy during training with these 3 metrics
        )

        # Print the model summary to the console
        full_model.summary()
        return full_model
    

    def update_base_model(self):
        # Update the base model to a full model with additional layers and configurations
        self.full_model = self._prepare_full_model(
            model=self.model,  # Pass the base model
            classes=self.config.params_classes,  # Number of classes from the config
            freeze_all=True,  # Freeze all layers during training
            freeze_till=None,  # No layers to unfreeze
            learning_rate=self.config.params_learning_rate  # Learning rate from the config
        )

        # Save the updated full model to the specified path
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the Keras model to the specified file path
        model.save(path)