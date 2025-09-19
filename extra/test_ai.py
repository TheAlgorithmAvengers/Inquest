import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 1. Load CSV file
data = pd.read_csv("/Users/sohamladda/Desktop/Inquest/extra/data.csv")

# 2. Split into features (X) and labels (y)
X = data.drop("label", axis=1).values  # features
y = data["label"].values               # labels (0 or 1)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a simple binary classification model
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),  # hidden layer
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # output layer for binary classification
])

# 5. Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Train the model
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# 7. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

  # shape must be (1, features)

model.save("model.h5")