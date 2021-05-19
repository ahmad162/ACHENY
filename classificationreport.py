import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
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

class ClassificationReport:
    @staticmethod
    def show(modelname='cheno',base_dir = 'I:/GHOLAMREZA_Z/ACHENY_LAST_64',testdir="test",fold=1):
        if modelname=='cheno':
            input_shape = (64, 64)
            test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True) 
        else:
            input_shape = (224, 224)
            test_datagen = ImageDataGenerator(rescale=1./255)
            base_dir=base_dir.replace("64","224")
        
        batch_size = 32

        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        CLASS_NO = len(os.listdir(train_dir))


        model = load_model(base_dir+'/'+modelname+'_'+str(fold)+'.h5')

        if testdir=="validation":
            dir_v=valid_dir
        elif testdir=="train":
            dir_v=train_dir
        else:
            dir_v=test_dir
        test_generator = test_datagen.flow_from_directory(dir_v, target_size=input_shape,
                        batch_size=batch_size,class_mode='categorical',shuffle=False)
        all_class=os.listdir(dir_v)
        test_loss, test_acc, top5_acc = model.evaluate_generator(test_generator)
        print('test acc:%.3f  test loss:%.3f'%(test_acc,test_loss))
        Y_pred = model.predict_generator(test_generator, np.ceil(test_generator.samples/test_generator.batch_size)) 
        y_pred = np.argmax(Y_pred, axis=1)
        print('Classification of ',dir_v,' Report')
        target_names = all_class 
        print(classification_report(test_generator.classes, y_pred, target_names=target_names))
        report = pd.DataFrame(list(precision_recall_fscore_support(test_generator.classes, y_pred)),
                index=['Precision', 'Recall', 'F1-score', 'Support']).T

        # Now add the 'Avg/Total' row
        report.loc['Avg/Total', :] = precision_recall_fscore_support(test_generator.classes, y_pred,average='weighted')
        report.loc['Avg/Total', 'Support'] = report['Support'].sum()
        report.loc[0:30,'']=target_names[0:30]
        report.to_csv(base_dir+'/'+modelname+'_'+testdir+'_'+str(fold)+'.csv', index= True)