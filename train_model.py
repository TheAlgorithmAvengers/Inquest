import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

import os

csv_path = os.path.join("resources", "resources.csv")

data = pd.read_csv(csv_path)

x = data.drop(columns=["label"]).values.astype('float64')

y = data["label"].values.astype('int')

y = to_categorical((y), num_classes=28)

print(x.shape[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(x.shape[1],)),  
    Dropout(0.2),
    Dense(128, activation='relu'),                                            
    Dropout(0.2),
    Dense(64, activation='relu'),                              
    Dense(28, activation='softmax') 
])

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',  
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25)

loss, accuracy = model.evaluate(x_test, y_test)

accuracy = int(accuracy * 100)

print(f"Test Accuracy: {accuracy}%")


model.save(f'trained_model.keras')
print(f"Model saved as trained_model.keras")




