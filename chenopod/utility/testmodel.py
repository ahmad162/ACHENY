import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from IPython.display import Image
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras import Model
import os
import scipy.io as sio
import efficientnet.keras as efn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix, precision_recall_fscore_support
import pandas as pd
from keras.models import load_model


class TestModel: #def __init__():       
    @staticmethod
    def test(modelname='cheno',test_dir = 'I:/GHOLAMREZA_Z/ACHENY_LAST_64/test',fold=1, batch_sz = 32):
        if modelname=='cheno':
            input_shape = (64, 64)
            test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True) 
        else:
            input_shape = (224, 224)
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_dir=test_dir.replace("64","224")
        
        
        test_generator = test_datagen.flow_from_directory(
            test_dir, 
            target_size=input_shape, 
            batch_size=batch_sz,
            class_mode='categorical',shuffle=False)
        model=load_model(test_dir[:test_dir.rfind('/')]+'/'+modelname+'_'+str(fold)+'.h5') 
        print(test_dir[test_dir.rfind('/'):]+' evaluate by '+modelname+'_'+str(fold)+'.h5')
        test_loss, test_acc, top5_acc = model.evaluate_generator(test_generator, steps=batch_sz)
        #print('%s acc:%.3f top5_acc:%.3f loss:%.3f\n'%(test_dir[test_dir.rfind('/')+1:],test_acc,top5_acc, test_loss))
        return test_loss, test_acc, top5_acc