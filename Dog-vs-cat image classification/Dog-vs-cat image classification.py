import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

trains = tf.keras.utils.image_dataset_from_directory(
    directory = "/kaggle/input/dogs-vs-cats/train",
    batch_size = 32,
    labels = "inferred",
    label_mode = "int",
    image_size = (256,256),
)
test = tf.keras.utils.image_dataset_from_directory(
    directory = "/kaggle/input/dogs-vs-cats/test",
    batch_size = 32,
    labels = "inferred",
    label_mode = "int",
    image_size = (256,256),
)


# normalize
def process(image,label):
    image = tf.cast(image/255,tf.float32)
    return image,label

train_data = trains.map(process)
test_data = test.map(process)


model = tf.keras.Sequential([
    layers.Conv2D(32, kernel_size = (3,3),padding = "same",activation="relu",input_shape=(256,256,3)),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size= (2,2),padding="same",strides = 2),
    layers.Conv2D(64,kernel_size=(3,3),padding="same",activation= "relu"),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),padding="same",strides = 2),

    layers.Conv2D(128,kernel_size=(3,3),padding="same",activation= "relu"),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),padding="same",strides = 2),
    
    layers.Conv2D(256,kernel_size=(3,3),padding="same",activation= "relu"),
    BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),padding="same",strides = 2),
    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),

])

model.compile(optimizer = "adam", loss ="binary_crossentropy", metrics = ["accuracy"])

history = model.fit(train_data,epochs = 1,validation_data=test_data)

img_path = "/kaggle/input/dogs-vs-cats/test/dogs/dog.1010.jpg"
img = image.load_img(img_path, target_size=(256, 256))  # adjust size to your model's input
from tensorflow.keras.preprocessing import image

imgarr = image.img_to_array(img)
imgarr = imgarr / 255.0 
imgarr = np.expand_dims(imgarr, axis=0)  # shape: (1, 32, 32, 3)

prediction = model.predict(imgarr)
predicted_class = int(prediction[0][0] > 0.5)
print("Predicted class:", "Dog" if predicted_class else "Cat")

