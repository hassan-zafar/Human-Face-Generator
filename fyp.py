from flask import Flask, request, jsonify, abort, session, flash, redirect, url_for, Response 
from keras.preprocessing.image import load_img
from tensorflow import keras
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

def predict_image():
    img = load_img('img.png', target_size=(64, 64))
    image = keras.preprocessing.image.img_to_array(img)
    image = image / 255.0
    image = image.reshape(1,64,64,3)
    model = tf.keras.models.load_model('pneumoniaModel')
    prediction = model.predict(image)
    print(prediction)
    if(prediction[0] > 0.5):
        stat = prediction[0] * 100 
        return "This image is %.2f percent %s"% (stat, "PNEUMONIA")
    else:
        stat = (1.0 - prediction[0]) * 100
        return "This image is %.2f percent %s" % (stat, "NORMAL")

@app.route('/predict', methods=['GET', 'POST'])
def process_predict():
	body = request.files
	body['image'].save('img.png')
	return { 'image': predict_image() }


if __name__ == '__main__':
    app.run(host='0.0.0.0')


