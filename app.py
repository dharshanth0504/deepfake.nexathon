import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the trained model
model_path = os.path.join(os.getcwd(), 'deepfake_detection_model.h5')
model = tf.keras.models.load_model(model_path)


# Function to predict if an image is real or fake
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    prediction = model.predict(img_array)
    print(f"Prediction Output: {prediction}")
    result = "Fake" if prediction[0][0] > 0.6 else "Real"

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            print("No file part")  # Debugging print
            return render_template("index.html", result="No file part")

        file = request.files["image"]

        if file.filename == "":
            print("No selected file")  # Debugging print
            return render_template("index.html", result="No file selected")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)  # Save file
            print(f"File saved at: {image_path}")  # Debugging print

            # Here, call the function to process the image
            result = predict_image(image_path)  # Your prediction function

            return render_template("index.html", result=result, filename="uploads/" + file.filename )

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)

