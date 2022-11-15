# load the trained CIFAR10 model
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models
from PIL import Image 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context                                                                               

import matplotlib.pyplot as plt
model = load_model('Group14_CIFARmodel.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img / 255.0
    return img
    
#https://stackoverflow.com/questions/72479044/cannot-import-name-load-img-from-keras-preprocessing-image
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
# get the image from the internet
URL = "https://static.toiimg.com/thumb/msid-67586673,width-1070,height-580,overlay-toi_sw,pt-32,y_pad-40,resizemode-75,imgsize-3918697/67586673.jpg"
# Cat 1      -- Challenge Test "https://static.toiimg.com/thumb/msid-67586673,width-1070,height-580,overlay-toi_sw,pt-32,y_pad-40,resizemode-75,imgsize-3918697/67586673.jpg"
# Cat 2      -- Challenge Test "https://wagznwhiskerz.com/wp-content/uploads/2017/10/home-cat.jpg"
# Bird 1     -- Challenge Test "https://ichef.bbci.co.uk/news/976/cpsprodpb/67CF/production/_108857562_mediaitem108857561.jpg"
# Bird 2     -- Challenge Test "https://upload.wikimedia.org/wikipedia/commons/5/53/Weaver_bird.jpg"
# Automobile -- Challenge Test "https://images.all-free-download.com/images/graphiclarge/classic_jaguar_210354.jpg" 

picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)
# show the picture
#image = plt.imread(picture_path)
#plt.imshow(image)
image = Image.open(picture_path)
image.show() 
# show prediction result.
print('\nPrediction: This image most likely belongs to ' + 
class_names[int(result.argmax(axis=-1))])

# # get the image from the internet
# URL = "https://image.shutterstock.com/image-vector/airplane-600w-646772488.jpg"
# picture_path  = tf.keras.utils.get_file(origin=URL)
# img = load_image(picture_path)
# result = model.predict(img)
# # show the picture
# image = plt.imread(picture_path)
# plt.imshow(image)
# # show prediction result.
# print('\nPrediction: This image most likely belongs to ' + 
# class_names[int(result.argmax(axis=-1))])
# ############################################################
# ## This website has everything to improve your accuracy  ###
# ############################################################
# # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 