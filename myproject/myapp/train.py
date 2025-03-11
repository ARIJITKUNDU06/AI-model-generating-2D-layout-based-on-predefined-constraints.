import tensorflow as tf
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("synthetic_layout_data.csv")
X_train = df.values  # Input is the same as output (autoencoder-like)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(30,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(30)  # Output same shape as input
])

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X_train, X_train, epochs=100, batch_size=32)

# Save
model.save("layout_model.h5")
