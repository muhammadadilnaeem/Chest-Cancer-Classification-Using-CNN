import os
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, render_template
from chest_cancer_classifier.utils.common_functions import decodeImage
from chest_cancer_classifier.pipeline.stage_5_prediction import PredictionPipeline

# Set environment variables for language settings
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for the app
CORS(app)

class ClientApp:
    def __init__(self):
        # Initialize with a default image filename
        self.filename = "inputImage.jpg"
        # Create an instance of the PredictionPipeline with the specified filename
        self.classifier = PredictionPipeline(self.filename)

# Route for the home page
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    # Render the index.html template for the home page
    return render_template('index.html')

# Route to trigger the training process
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    # Execute the training script (demo.py)
    os.system("python demo.py")
    # Uncomment the line below if using DVC for reproducible pipelines
    # os.system("dvc repro")
    return "Training done successfully!"

# Route to handle image prediction
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Extract the image data from the POST request
    image = request.json['image']
    # Decode the image and save it using the filename from ClientApp
    decodeImage(image, clApp.filename)
    # Use the classifier to predict the class of the image
    result = clApp.classifier.predict()
    # Return the prediction result as a JSON response
    return jsonify(result)

# Main entry point for the application
if __name__ == "__main__":
    # Create an instance of ClientApp
    clApp = ClientApp()
    # Run the Flask application on all interfaces, port 8080 (suitable for AWS)
    app.run(host='0.0.0.0', port=8080)