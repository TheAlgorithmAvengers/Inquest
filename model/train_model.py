import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers

import tensorflow as tf

import os

csv_path = os.path.join("resources", "resources2.csv")

data = pd.read_csv(csv_path)

data = data[data["label"] != 28]

x = data.drop(columns=["label"]).values.astype('float64')

y = data["label"].values.astype('int')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4, stratify=y)
y_train = to_categorical(y_train, num_classes=30)
y_test = to_categorical(y_test, num_classes=30)

normalizer = layers.Normalization()
normalizer.adapt(x_train)

model = Sequential([
    normalizer,
    Dense(256, activation='relu'),                                              
    Dropout(0.3),
    Dense(128, activation='relu'),                                              
    Dropout(0.3),
    Dense(64, activation='relu'),                              
    Dense(30, activation='softmax') 
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',  
    metrics=['accuracy'],
)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.25, callbacks=[early_stopping])

loss, accuracy = model.evaluate(x_test, y_test)

accuracy = int(accuracy * 100)

print(f"Test Accuracy: {accuracy}%")


model.save(f'trained_model_{accuracy}.keras')
print(f"Model saved as trained_model_{accuracy}.keras")




