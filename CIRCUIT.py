import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from flask import Flask, request, jsonify

# ✅ **Step 1: Generate Synthetic Data with 30 Features**
def generate_synthetic_data(num_samples=1000, board_size=(10, 10)):
    data = []
    labels = []
    for _ in range(num_samples):
        num_components = random.randint(3, 10)
        components = []
        for _ in range(num_components):
            x, y = random.randint(0, board_size[0]-1), random.randint(0, board_size[1]-1)
            power = random.uniform(0.5, 5.0)
            components.append((x, y, power))
        label = sorted(components, key=lambda c: c[2], reverse=True)  # Optimize placement by power
        data.append(components)
        labels.append(label)
    return np.array(data, dtype=object), np.array(labels, dtype=object)

data, labels = generate_synthetic_data(500)

# ✅ **Step 2: Convert Data to 30 Features for Model Compatibility**
max_components = 10  # Ensuring 30 features when flattened

def pad_or_trim(components, max_length):
    """Pads with (-1, -1, -1) or trims to ensure a fixed size"""
    if len(components) < max_length:
        components += [(-1, -1, -1)] * (max_length - len(components))  # Pad
    return components[:max_length]  # Trim if necessary

X_train = np.array([np.array(pad_or_trim(d, max_components)).flatten() for d in data], dtype=float)
y_train = np.array([np.array(pad_or_trim(l, max_components)).flatten() for l in labels], dtype=float)

# ✅ **Step 3: Define & Load Neural Network Model**
model = Sequential([
    Dense(64, activation='relu', input_shape=(30,)),  # Ensuring input shape is (30,)
    Dense(128, activation='relu'),
    Dense(30, activation='linear')  # Output layer with same shape as input
])

# ✅ **Load the trained model correctly**
model = tf.keras.models.load_model(
    "D:/Backup File/INSYDE.IO/layout_model.h5",
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

# ✅ **Step 4: Train Model**
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# ✅ **Step 5: Save Model**
model.save("layout_model.h5")

# ✅ **Step 6: Visualization Function**
def visualize_layout(original, predicted):
    plt.figure(figsize=(6, 6))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.scatter([c[0] for c in original], [c[1] for c in original], c='blue', label='Original')
    plt.scatter([c[0] for c in predicted], [c[1] for c in predicted], c='red', marker='x', label='Predicted')
    plt.legend()
    plt.title('Optimized Component Placement')
    plt.show()

# ✅ **Step 7: Flask API for Deployment**
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["layout"]
    input_data = np.array(data).flatten()
    
    # ✅ **Ensure input always has exactly 30 features**
    required_features = 30
    if len(input_data) < required_features:
        input_data = np.pad(input_data, (0, required_features - len(input_data)), mode='constant', constant_values=-1)
    else:
        input_data = input_data[:required_features]

    # ✅ **Reshape input to (1, 30)**
    input_data = input_data.reshape(1, required_features)

    # ✅ **Debugging Step Before Prediction**
    print(f"✅ Input shape before prediction: {input_data.shape}")

    # ✅ **Ensure Model Prediction Works with Correct Shape**
    prediction = model.predict(input_data)

    # ✅ **Reshape Output to Match Expected Format (x, y, power)**
    predicted_layout = prediction.reshape(-1, 3).tolist()

    # ✅ **Visualizing Predicted Layout**
    visualize_layout(data, predicted_layout)

    return jsonify({"optimized_layout": predicted_layout})

if __name__ == '__main__':
    app.run(debug=True)
