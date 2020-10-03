import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

train_data = pd.read_csv("../data_csv/train.csv")
test_data = pd.read_csv("../data_csv/test.csv")

# Split train to x and y
y = train_data["label"]
del train_data["label"]

# Scale data
train_data = train_data/255
test_data = test_data/255

# Reshape image to match Keras
train_data = train_data.values.reshape(-1, 28, 28, 1)
test_data = test_data.values.reshape(-1, 28, 28, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=1)

# Encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=80, epochs=5, validation_data=(X_test, y_test), verbose=1)

# Fit unlabeled data
Y = pd.Series(np.argmax(model.predict(test_data), axis=1))
Y.name = "Label"
image_id = pd.Series(range(1, Y.shape[0] + 1))
image_id.name = "ImageId"
prediction_data = pd.concat([image_id, Y], axis=1)
# Create prediction csv
prediction_data.to_csv("../data_csv/prediction.csv", index=False)
