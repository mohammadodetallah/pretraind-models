import requests
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input as mo
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import  decode_predictions , ResNet50
from keras.applications.resnet50 import preprocess_input as net
from keras.applications.mobilenet import preprocess_input , decode_predictions , MobileNet
from keras.applications.inception_v3 import preprocess_input , decode_predictions , InceptionV3
from keras.applications.xception import preprocess_input , decode_predictions , Xception
from keras.applications.densenet import decode_predictions , DenseNet121
from keras.applications.densenet import preprocess_input as den
from keras.applications.nasnet import decode_predictions , NASNetLarge
from keras.applications.nasnet import preprocess_input as nas
import numpy as np
import json

def vgg16(image):
    model = VGG16()
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = mo(image)
    predection = model.predict(image)
    preddected = decode_predictions(predection)
    preddected = preddected[0][0]
    preddected = {"model": 'vgg16', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def resnet50(image):
    model = ResNet50(weights='imagenet')
    image = np.expand_dims(image, axis=0)
    image = net(image)
    predection = model.predict(image)
    preddected = decode_predictions(predection)
    preddected = preddected[0][0]

    preddected = {"model": 'resnet5', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def mobileneet(image):
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model=MobileNet()
    predection = model.predict(image)
    preddected = imagenet_utils.decode_predictions(predection)
    preddected = preddected[0][0]

    preddected = {"model": 'mobileneet', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def inceptionv3(image):
    file_name = "object"
    image = load_img(file_name, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model = InceptionV3(weights='imagenet')
    predection = model.predict(image)
    preddected = imagenet_utils.decode_predictions(predection)
    preddected = preddected[0][0]

    preddected = {"model": 'inceptionv3', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def densenet(image):
    image = np.expand_dims(image, axis=0)
    image = den(image)
    model = DenseNet121(weights='imagenet')
    predection = model.predict(image)
    preddected = imagenet_utils.decode_predictions(predection)
    preddected = preddected[0][0]

    preddected={"model":'densenet' ,"object_detected":preddected [1],"accurecy" :preddected[2]*100}
    return  preddected
def nasenet(image):
    file_name='object'
    image = load_img(file_name, target_size=(331, 331))
    image = np.expand_dims(image, axis=0)
    image = nas(image)
    model = NASNetLarge()
    predection = model.predict(image)
    preddected = imagenet_utils.decode_predictions(predection)
    preddected = preddected[0][0]

    preddected = {"model": 'nasenet', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def xception(image):
    file_name = "object"
    image = load_img(file_name, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model = Xception()
    predection = model.predict(image)
    preddected = imagenet_utils.decode_predictions(predection)
    preddected = preddected[0][0]

    preddected = {"model": 'xception', "object_detected": preddected[1], "accurecy": preddected[2]*100}
    return preddected
def pre_procces_image(url):
    file_name= "object"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)
    image = load_img(file_name, target_size=(224, 224))
    image = img_to_array(image)
    return image
def get_accurecy(detected):
    return detected.get('accurecy')


image_url="https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234558/Chinook-On-White-03.jpg"

image = pre_procces_image(image_url)
predected = []
predected.append(vgg16(image))
predected.append(resnet50(image))
predected.append(mobileneet(image))
predected.append(inceptionv3(image))
predected.append(xception(image))
predected.append(densenet(image))
predected.append(nasenet(image))
predected.sort(key=get_accurecy , reverse=True)
print(predected,  end='\n\n')
json.dumps(predected)





