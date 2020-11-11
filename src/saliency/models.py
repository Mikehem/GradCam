# Models architecture
import os
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.applications.resnet50 import ResNet50

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

MODEL_DIR = "../models"


class VanilaResNet50:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def __call__(self):
        resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')
        for layer in resnet.layers:
            layer.trainable = False

        logits = Dense(2)(resnet.layers[-1].output)
        output = Activation('softmax')(logits)
        model = Model(resnet.input, output)

        return model


class ResNet50PlusFC:
    def __init__(self, n_classes=2):
        resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')
        for layer in resnet.layers:
            layer.trainable = False

        fc1 = Dense(100)(resnet.layers[-1].output)
        fc2 = Dense(100)(fc1)
        logits = Dense(2)(fc2)
        output = Activation('softmax')(logits)
        model = Model(resnet.input, output)

        return model


def load_VanilaResNet50():
    resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False

    logits = Dense(2)(resnet.layers[-1].output)
    output = Activation('softmax')(logits)
    model = Model(resnet.input, output)
    model.load_weights("{}/resnet50best.hdf5".format(MODEL_DIR))

    return model

def load_ResNet50PlusFC():
    resnet = ResNet50V2(include_top=False, pooling="avg", weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False

    fc1 = Dense(100)(resnet.layers[-1].output)
    fc2 = Dense(100)(fc1)
    logits = Dense(2)(fc2)
    output = Activation('softmax')(logits)
    model = Model(resnet.input, output)
    model.load_weights("{}/resnet50fcbest.hdf5".format(MODEL_DIR))

    return model

def load_ResNet50():
    return ResNet50(include_top=True, weights='imagenet')

def load_ResNet50V2():
    return ResNet50V2(include_top=True, weights='imagenet')
