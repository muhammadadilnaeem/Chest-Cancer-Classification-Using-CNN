
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

class PredictionPipeline:
    def __init__(self, filename):
        # Initialize the pipeline with the image filename
        self.filename = filename

    def predict(self):
        # Load the pre-trained model from the specified path
        model = load_model(os.path.join("model", "model.h5"))

        # Load the image and resize it to match the model's input size (224x224)
        test_image = load_img(self.filename, target_size=(224, 224))
        
        # Convert the image to a numpy array
        test_image = img_to_array(test_image)
        
        # Expand dimensions to create a batch of one image
        test_image = np.expand_dims(test_image, axis=0)
        
        # Use the model to predict the class of the image
        result = np.argmax(model.predict(test_image), axis=1)
        
        # Output the prediction result
        print(result)

        # Check the predicted class and return the corresponding label
        if result[0] == 1:
            prediction = 'Normal'
            return [{"image": prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{"image": prediction}]




# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# class PredictionPipeline:
#     def __init__(self, filename):
#         # Initialize the pipeline with the image filename
#         self.filename = filename

#     def predict(self):
#         # Load the pre-trained model from the specified path
#         # Uncomment the line below if using a different model path
#         # model = load_model(os.path.join("artifacts", "training", "model.h5"))
#         model = load_model(os.path.join("model", "model.h5"))

#         # Store the image name from the initialized filename
#         imagename = self.filename
        
#         # Load the image and resize it to match the model's input size (224x224)
#         test_image = image.load_img(imagename, target_size=(224, 224))
        
#         # Convert the image to a numpy array
#         test_image = image.img_to_array(test_image)
        
#         # Expand dimensions to create a batch of one image
#         test_image = np.expand_dims(test_image, axis=0)
        
#         # Use the model to predict the class of the image
#         result = np.argmax(model.predict(test_image), axis=1)
        
#         # Output the prediction result
#         print(result)

#         # Check the predicted class and return the corresponding label
#         if result[0] == 1:
#             prediction = 'Normal'
#             return [{"image": prediction}]
#         else:
#             prediction = 'Adenocarcinoma Cancer'
#             return [{"image": prediction}]