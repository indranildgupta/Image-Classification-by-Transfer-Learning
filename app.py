from flask import Flask,render_template,request
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
from keras.applications import VGG19

app = Flask(__name__)

conv_vgg = VGG19(include_top=False,input_shape=(224, 224, 3),
                weights='imagenet')

model=load_model("model1.h5")

def load_image(img_path):
    org_img = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) 
    img_tensor /= 255.  
    # plt.imshow(org_img)                           
    # plt.axis('off')
    # plt.show()
    
    return img_tensor

@app.route('/',methods=['GET'])
def hello_word():
    return render_template ('index.html')


@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']  #same as HTML input file name
        image_path = "./images/" + imagefile.filename 
        imagefile.save(image_path)
    except:
        return render_template ('index.html')

    img_tensor = load_image(image_path)
    features = conv_vgg.predict(img_tensor.reshape(1,224, 224, 3)) #default input shape of VGG model 

    # Make prediction
    try:
        prediction = model.predict_classes(features)
    except:
        prediction = model.predict_classes(features.reshape(1, 7*7*512))
        
    classes = ["Buildings", "Forest", "Glacier", "Mountains", "Sea", "Street"]
    res = classes[prediction[0]]
    os.remove(image_path)
    return render_template ('index.html', prediction=res)

if __name__ == '__main__':
    app.run(port=3000,debug=True)