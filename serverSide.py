#set FLASK_APP=serverSide.py
#set FLASK_ENV=development
#flask run --host=192.168.43.18
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import load_model
from flask import Flask,jsonify,request
import base64
import io
from keras.preprocessing.image import img_to_array
import numpy as np
import time
from keras.preprocessing.image import save_img
from numpy.core.numeric import outer
import tensorflow as tf
from PIL import Image
from flask_restful import Resource,Api,reqparse
app=Flask(__name__)
api =Api(app)
import os
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub Version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

model =load_model('fully_trained.h5')
print('Model Loaded')
outputfile='output.png'
savePath='./output/'
def prepare_image(image,target):
    if image.mode !="RGB":
        image=image.convert('RGB')
     
    image=image.resize(target)
    image=img_to_array(image)
    image=(image-127.5)/127.5
    image =np.expand_dims(image,axis=0)
    return image
def predict_image(prepared_image):
    preds=model.predict(prepared_image)

   

    output=tf.reshape(preds,[256,256,3])
    output = (output+1)/2
    save_img(savePath+outputfile,img_to_array(output))
class Predict(Resource):
    def post(self):
        json_data=request.get_json()
        img_data=json_data['image']

        image=base64.b64decode(str(img_data))
        
        img=Image.open(io.BytesIO(image))
        encoded_string=[]
        prepared_image=prepare_image(img,target=(256,256))
        for x in range(4):
            predict_image(prepared_image)
            imageNew=Image.open(savePath+outputfile)
            imageNew=imageNew.resize((50,50))
            imageNew.save(savePath+'new_'+x+outputfile)

            with open(savePath+'new_'+x+outputfile,'rh') as image_file:
                encoded_string[x]=base64.b64encode(image_file.read())

        outputData={
            'image1':str(encoded_string[0]),
            'image2':str(encoded_string[1]),
            'image3':str(encoded_string[2]),
            'image4':str(encoded_string[3]),

        }
        return outputData


api.add_resource(Predict,'/predict')

if __name__=='__main__':
    app.run(debug=False)