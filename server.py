import boto3
import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import io
from model import FakeImageModel
import traceback
import torch.nn.functional as F  # For softmax function

# init app
app = Flask(__name__)

# aws config
s3_client = boto3.client('s3')
BUCKET_NAME = 'fakedetection-bucket'
REGION = 'ap-northeast-1'  


# configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# device configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# load trained model
model = FakeImageModel().to(device)
model.load_state_dict(torch.load('model.pth'))  # Load the trained weights
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_valid_image(file):
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.verify()  # Verify if it's a valid image
        return True
    except (IOError, SyntaxError):
        return False

@app.route('/')
def index():
    return render_template('index.html')  # Serve the frontend HTML page

# endpoint - image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save img to BytesIO buffer
        img = Image.open(file.stream)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)  # Ensure the pointer is at the start of the buffer
        img = transform(img).unsqueeze(0) 

        # Save img to S3
        try:
            s3_key = file.filename
            s3_client.upload_fileobj(img_bytes, BUCKET_NAME, s3_key)
            print(f"Successfully uploaded {file.filename} to S3.")
        except Exception as e:
            print(f"Error uploading image to S3: {str(e)}")
            return jsonify({'error': f'Error uploading image to S3: {str(e)}'}), 500
        
        # Make prediction with model
        with torch.no_grad():
            output = model(img)
            tempature = 1.0
            probabilities = F.softmax(output / tempature, dim=1)  # Get probabilities (confidence)
            _, predicted = torch.max(probabilities, 1)  # Get the predicted class
            
        # Get the confidence value
        confidence = probabilities[0, predicted].item()  # Extract confidence from the softmax output
        
        # Convert prediction to human-readable label
        prediction = 'real' if predicted.item() == 0 else 'fake'
        
        # Construct S3 URL
        s3_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{s3_key}"

        # Return prediction, confidence, and image path
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,  # Include the confidence
            'img_url': s3_url
        }), 200
    
    except Exception as e:
        print(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
