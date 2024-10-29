from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

# Path to upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(50, 50))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    return pred

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Check if the file is valid
        if f and f.filename:
            # Secure the filename and save it to the upload folder
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                f.save(file_path)  # Save the uploaded file
                print(f'File saved at: {file_path}')  # Debugging statement

                # Make prediction
                pred = model_predict(file_path, model)

                # Remove the saved file after prediction (if desired)
                # os.remove(file_path) 

                # Arrange the correct return according to the model
                # Arrange the correct return according to the model
                prediction_class = 'Malaria Parasitized' if pred[0] == 0 else 'Normal'

                # Symptoms and precautions (only for "Malaria Parasitized")
                if prediction_class == 'Malaria Parasitized':
                    precautions = [
                        "Consult a healthcare professional immediately.",
                        "Take prescribed anti-malarial medication as directed.",
                        "Use mosquito nets and insect repellents to prevent further bites.",
                        "Stay indoors during peak mosquito activity hours (dawn and dusk).",
                        "Ensure that living areas are well-screened against mosquitoes.",
                        "Stay hydrated and rest to help recovery."
                    ]
                    symptoms = [
                        "Fever and chills.",
                        "Flu-like symptoms, including headache and fatigue.",
                        "Sweats.",
                        "Nausea and vomiting.",
                        "Muscle pain.",
                        "Anemia (low red blood cell count).",
                        "Respiratory distress in severe cases."
                    ]
                else:
                    precautions = None
                    symptoms = None

                # Return the result page
                return render_template('result.html', 
                                    prediction_class=prediction_class, 
                                    precautions=precautions, 
                                    symptoms=symptoms, 
                                    uploaded_image=filename)

            except Exception as e:
                print(f'Error saving file: {e}')  # Debugging statement
                return "Error saving file."
        else:
            return "No file uploaded or file is invalid."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
