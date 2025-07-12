"""import pandas
import numpy
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout
import pickle

#generators
train_ds = keras.utils.image_dataset_from_directory(
    directory='data',
    labels="inferred",#label of image automatically determine by the use of subdirectory name
    label_mode='int',
    batch_size=32,
    image_size=(300,300)# it reshape the image by 256,256
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='data',
    labels="inferred",#label of image automatically determine by the use of subdirectory name
    label_mode='int',
    batch_size=32,
    image_size=(300,300)# it reshape the image by 256,256
)

# normalize
def process(image,label):
    image=tf.cast(image/255. ,tf.float32)
    return image,label

train_ds= train_ds.map(process) #map function ek ek krke image bhejega and wo normalize hoke wapis aaige
validation_ds=validation_ds.map(process)

#creat cnn model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(300,300,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5,activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()


"""



import pandas
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
import pickle

# Dataset loading
train_ds = keras.utils.image_dataset_from_directory(
    directory='data',
    labels="inferred",
    label_mode='int',  # Integer labels
    batch_size=32,
    image_size=(300, 300)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='data',
    labels="inferred",
    label_mode='int',
    batch_size=32,
    image_size=(300, 300)
)

# Normalize
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(300, 300, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))  # 5 classes

model.summary()

# Compile using sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_ds, epochs=2, validation_data=validation_ds)

# Plot accuracy
plt.plot(history.history['accuracy'], color='red', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], color='red', label='Train Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
plt.legend()
plt.show()

with open('data.pkl', 'wb') as file:
    pickle.dump(model, file)