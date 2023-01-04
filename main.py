import tensorflow as tf  #import machine learning package

savedModel = tf.keras.models.load_model("OOP_model.h5") #load saved model

# Load and preprocess the image
#"car.jpg" is name of my random testing image from internet, you can import your own image to same folder as python file and try it 

def prediction(image_name):
    image = tf.keras.preprocessing.image.load_img(image_name, target_size=(32, 32))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image = image[tf.newaxis, ...]  # Add an extra dimension for the batch size

    # Predict the class of the image
    prediction = savedModel.predict(image)

    # Get the index of class with the highest probability
    class_idx = tf.argmax(prediction[0]).numpy()

    # decision algo for printing name of class with specified index
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            
    print(f"Class index is {class_idx} and the class name is {classes[class_idx]}!")


prediction("car.jpg")

