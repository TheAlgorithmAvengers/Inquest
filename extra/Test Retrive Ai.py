from tensorflow import keras

# Load the model
model = keras.models.load_model("model.h5")

# Use it directly for predictions
import numpy as np
new_data = np.array([[20, 15, 10]])
probability = model.predict(new_data)
prediction = (probability > 0.5).astype("int32")

print("Predicted probability:", probability[0][0])
print("Predicted class:", prediction[0][0])