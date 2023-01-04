import tensorflow as tf

#Load and preprocess the data
#cifar-10 is training dataset with 10 classes of pictures, each class contain 6000 pictures, together it is 60k pictures
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

class Neural_network:
    def __init__(self, count_of_epochs) -> None:
        self.count_of_epochs = count_of_epochs

    def training_model(self):  #this function trains the model

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
        model.fit(x_train, y_train, epochs = self.count_of_epochs, validation_data=(x_test, y_test))

        #Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        #save model
        model.save("OOP_model.h5")
        print("Model saved!")

network = Neural_network(1)  #create new instance of class Neural_network with specific count of traing epochs
network.training_model()    #train the model
savedModel = tf.keras.models.load_model("OOP_model.h5")  #save model to file



