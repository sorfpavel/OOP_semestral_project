import tensorflow as tf

class Picture:   #class for object picture creation

    def __init__(self, image_name):   #initializition with name of the image for recognition
        self.image_name = image_name

    def recognition(self):   #function for image recognition
        savedModel = tf.keras.models.load_model("OOP_model.h5")  #load saved model
        image = tf.keras.preprocessing.image.load_img(self.image_name, target_size=(32, 32))  #preprocess the image
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        image = image[tf.newaxis, ...]  # Add an extra dimension for the batch size

        # Predict the class of the image
        model = savedModel.predict(image)

        # Get the index of class with the highest probability
        class_idx = tf.argmax(model[0]).numpy()

        # decision algo for printing name of class with specified index
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                
        print(f"Class index is {class_idx} and the class name is {classes[class_idx]}!")