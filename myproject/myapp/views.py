import tensorflow as tf
from keras.losses import MeanSquaredError
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from .models import UserProfile

# Load the trained model
model = tf.keras.models.load_model(
    "D:/Backup File/INSYDE.IO/layout_model.h5",
    custom_objects={'mse': MeanSquaredError()}
)

# Home View
def home(request):
    return HttpResponse("Welcome to my Django project!")

# User List View
def user_list(request):
    return HttpResponse("Welcome to the Users Page!")

# Function to visualize the layout
def plot_layout(predicted_layout):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Optimized Circuit Board Layout")

    for (x, y, power) in predicted_layout:
        ax.scatter(x, y, s=power * 20, label=f"Power: {power:.2f}")

    ax.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return encoded_img

# Function to serve decoded Base64 image
def get_layout_image(request):
    base64_data = request.GET.get("image_data", "")  # Get Base64 string from request
    if not base64_data:
        return JsonResponse({"error": "No image data provided"}, status=400)

    try:
        image_data = base64.b64decode(base64_data)
        return HttpResponse(image_data, content_type="image/png")
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# Predict View
def predict(request):
    if request.method == "GET":
        input_data = np.random.rand(1, 30)  # Assuming 10 components with (x, y, power)

        prediction = model.predict(input_data)

        print("Raw Prediction Shape:", prediction.shape)  # Debugging
        
        if prediction.shape[1] != 30:
            return JsonResponse({"error": "Unexpected model output shape"}, status=500)

        # Reshape prediction into (10, 3) meaning (x, y, power) for 10 components
        predicted_layout = np.array(prediction).reshape(10, 3)

        # Apply constraints
        predicted_layout[:, :2] = np.clip(predicted_layout[:, :2], 0, 10)  # (x, y) within board
        predicted_layout[:, 2] = np.clip(predicted_layout[:, 2], 0.5, 5.0)  # power constraints

        img_str = plot_layout(predicted_layout)

        return JsonResponse({
            "prediction": predicted_layout.tolist(),
            "layout_image": img_str  # Base64 encoded image
        })
