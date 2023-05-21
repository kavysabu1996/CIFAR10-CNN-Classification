import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import argparse
import numpy as np

(_,_),(x_test,y_test) = cifar10.load_data()

labels = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer",
          5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}


model = tf.keras.models.load_model('src/models/cnn_epoch10.h5')
x_test  = x_test/255 

def make_prediction(idx):
    pred = model.predict(x_test[idx][np.newaxis, ...],verbose=0)
    return np.argmax(pred)

def plot_img(idx):
    prediction = make_prediction(idx)
    predicted_label = labels[prediction]
    correct_label = labels[y_test[idx][0]]
    if predicted_label==correct_label:
        print("{:*^80}".format('Correct Prediction'))
    else:
        print("{:x^80}".format("Wrong Prediction"))
    print("predicted label : {}".format(predicted_label))
    plt.imshow(x_test[idx])
    print("original label : {}".format(correct_label))
    plt.axis('off')
    plt.show()

parser = argparse.ArgumentParser("plotting result")
parser.add_argument('--idx', type=int)

args = parser.parse_args()

plot_img(args.idx)
plt.show()