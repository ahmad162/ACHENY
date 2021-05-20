import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Activation, BatchNormalization, Conv2D, MaxPooling2D
import numpy as np
from IPython.display import Image
from keras import optimizers
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import  ModelCheckpoint, EarlyStopping

import os
import matplotlib.pyplot as plt
import json

class ChenoScratch:
    @staticmethod
    def build(base_dir = "I:/GHOLAMREZA_Z/ACHENY_LAST_64", EPOCHS=200, patient=50,fold=1):
        input_shape = (64,64)
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        CLASS_NO = len(os.listdir(train_dir))
        BATCH_SZ = 32 #128
        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(*input_shape,3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(CLASS_NO, activation='softmax'))
        
        opt_rms = keras.optimizers.adam(lr=1e-3,decay=1e-6)  
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms,metrics=['accuracy','top_k_categorical_accuracy']) 
        
        train_datagen=ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True,
                                  rotation_range=40,width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')                              

        test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)
                               
        train_generator=train_datagen.flow_from_directory(train_dir, target_size=input_shape,batch_size=BATCH_SZ,
                                                  class_mode='categorical',shuffle=True)

        validation_generator=test_datagen.flow_from_directory(valid_dir,target_size=input_shape, 
                                                      batch_size=BATCH_SZ, class_mode='categorical')  

        weights_dir = base_dir+'/cheno_weight_'+str(fold)+'.h5'
        if os.path.exists(weights_dir):
                model.load_weights(weights_dir)
                print(weights_dir,' loaded')
        else:
            print(weights_dir,' will be create')
        checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=patient) 
        
        step_size_train=int(np.ceil(train_generator.n/train_generator.batch_size))
        step_size_validation=int(np.ceil(validation_generator.n/validation_generator.batch_size))
        def lr_schedule(epoch):
            lrate = 0.001
            if epoch > 50:
                lrate = 0.0005
            if epoch > 75:
                lrate = 0.0003
            return lrate
        history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            epochs=EPOCHS,
                            validation_data=validation_generator,
                            validation_steps=step_size_validation,
                            workers=8,             
                            max_queue_size=32,             
                            verbose=1,
                            callbacks=[checkpointer, early_stopping,LearningRateScheduler(lr_schedule)])
        acc=history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        top5_acc=history.history['top_k_categorical_accuracy']
        val_top5_acc=history.history['val_top_k_categorical_accuracy']
        with open(base_dir+'/cheno_'+str(fold)+'.txt', 'w') as filehandle: 
            json.dump((acc,val_acc,loss,val_loss,top5_acc,val_top5_acc), filehandle)
        model.save(base_dir+'/cheno_'+str(fold)+'.h5')
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b.-', label='Training acc')
        plt.plot(epochs, val_acc, 'g', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy', 'valid_accuracy'], loc='best')
        plt.figure()
        plt.plot(epochs, loss, 'b.-', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'valid_loss'], loc='best')
        plt.figure()
        plt.plot(epochs, top5_acc, 'b.-', label='Training top5-acc')
        plt.plot(epochs, val_top5_acc, 'g', label='Validation top5-acc')
        plt.title('Training and validation Top5 accuracy')
        plt.ylabel('top5 accuracy')
        plt.xlabel('epoch')
        plt.legend(['top5_acc', 'valid_top5_acc'], loc='best')
        plt.show()
        
        batch_sz=32

        for tested_dir in [test_dir, train_dir, valid_dir] :
            test_generator = test_datagen.flow_from_directory(
                tested_dir, 
                target_size=input_shape, 
                batch_size=BATCH_SZ,
                class_mode='categorical',shuffle=False)

            test_loss, test_acc, top5_acc = model.evaluate_generator(test_generator)#, steps=batch_sz)
            print('%s acc:%.3f top5_acc:%.3f loss:%.3f\n'%(tested_dir[tested_dir.rfind('/')+1:],test_acc,top5_acc, test_loss))
            