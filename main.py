import tensorflow as tf  #import machine learning package

#Load and preprocess the data
#cifar-10 is training dataset with 10 classes of pictures, each class contain 6000 pictures, together it is 60k pictures
#50k is used for training, 10k is used for test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()


#Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Reshape(target_shape=(32, 32, 3), input_shape=(32, 32, 3)),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=10, activation='softmax')
])


#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
#15 training epochs gave me about 80% accuracy
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

#Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Load and preprocess the image
#"image2.jpg" is name of my random testing image from internet, you can import your own omage to jupyterlab and try it 
image = tf.keras.preprocessing.image.load_img('image2.JPG', target_size=(32, 32))
image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
image = image[tf.newaxis, ...]  # Add an extra dimension for the batch size

# Predict the class of the image
prediction = model.predict(image)

# Get the index of class with the highest probability
class_idx = tf.argmax(prediction[0]).numpy()

# decision algo for printing name of class with specified index
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
print(f"Class index is {class_idx} and the class name is {classes[class_idx]}")

