from tensorflow import keras

# Load the model
model = keras.models.load_model("my_model.keras")

# Use it directly for predictions
import numpy as np
new_data = np.array([[67,69,420]])
probability = model.predict(new_data)
prediction = (probability > 0.5).astype("int32")

print("Predicted probability:", probability[0][0])
print("Predicted class:", prediction[0][0])