import keras
from keras.applications import VGG16
import os, json
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
import matplotlib.pyplot as plt
from keras.callbacks import  ModelCheckpoint, EarlyStopping
import gc

class Vgg16Acheny:
    @staticmethod
    def build(base_dir = "I:/GHOLAMREZA_Z/ACHENY_LAST_224", EPOCHS=200, patient=3,fold=1):
        input_shape = (224,224)
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        CLASS_NO = len(os.listdir(train_dir))
        BATCH_SZ = 32 #128

        conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(*input_shape, 3)) 
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(CLASS_NO, activation='softmax'))
        conv_base.trainable=False
        model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizers.RMSprop(lr=1e-5),
                      metrics=['acc','top_k_categorical_accuracy'])
        train_datagen = ImageDataGenerator(rescale=1./255,
              rotation_range=40, width_shift_range=0.2,
              height_shift_range=0.2,shear_range=0.2,
              zoom_range=0.2,horizontal_flip=True,
              fill_mode='nearest')

        # validation data should not be augmented
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                train_dir, target_size=input_shape, 
                batch_size=BATCH_SZ,class_mode='categorical') 

        validation_generator = test_datagen.flow_from_directory(
                valid_dir, target_size=input_shape, 
                batch_size=BATCH_SZ,class_mode='categorical') 

        weights_dir = base_dir+'/VGG16_weight_'+str(fold)+'.h5'

        checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=patient)

        history = model.fit_generator(train_generator,
                  steps_per_epoch=np.ceil(train_generator.samples/train_generator.batch_size),
                  epochs=EPOCHS, workers=8,             
                  max_queue_size=32, verbose=1,
                  validation_data=validation_generator,
                  validation_steps=np.ceil(validation_generator.samples/validation_generator.batch_size),
                  callbacks=[checkpointer, early_stopping])

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        top5_acc = history.history['top_k_categorical_accuracy']
        val_top5_acc = history.history['val_top_k_categorical_accuracy']

        conv_base.trainable = True
        for layeh in conv_base.layers[1:]:
              layeh.trainable = False

        k=13  
        for layer in range(18,1,-1):  #0
            if not conv_base.layers[layer].name.endswith('pool'):
                gc.collect()
                print("*** Layer[%d:] Training ***"%(k))
  
                if os.path.exists(weights_dir):
                     model.load_weights(weights_dir) 
            
                for layeh in conv_base.layers[layer:]:
                     layeh.trainable = True

                model.compile(loss='categorical_crossentropy', 
                              optimizer=optimizers.RMSprop(lr=1e-5),
                              metrics=['acc','top_k_categorical_accuracy'])
      
                checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
                early_stopping = EarlyStopping(patience=patient)

                history = model.fit_generator(
                      train_generator,
                      steps_per_epoch=np.ceil(train_generator.samples/train_generator.batch_size), 
                      epochs=EPOCHS, workers=8,max_queue_size=32,verbose=1,
                      validation_data=validation_generator,
                      validation_steps=np.ceil(validation_generator.samples/validation_generator.batch_size),
                      callbacks=[checkpointer, early_stopping]) 
    
                acc = acc + history.history['acc']
                val_acc = val_acc + history.history['val_acc']
                loss = loss + history.history['loss']
                val_loss = val_loss + history.history['val_loss']
                top5_acc = top5_acc + history.history['top_k_categorical_accuracy']
                val_top5_acc = val_top5_acc + history.history['val_top_k_categorical_accuracy']
                k=k-1
        
                with open(base_dir+'/VGG16_'+str(fold)+'.txt', 'w') as filehandle:  
                    json.dump((acc,val_acc,loss,val_loss,top5_acc,val_top5_acc), filehandle)
  
                print('VGG16_'+str(fold)+'.txt & ','VGG16_'+str(fold)+'.h5  saved ...')

                model.save(base_dir+'/VGG16_'+str(fold)+'.h5')

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b.-', label='Training acc')
        plt.plot(epochs, val_acc, 'g', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b.-', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.figure()

        plt.plot(epochs, top5_acc, 'b-', label='top5 accuracy')
        plt.plot(epochs, val_top5_acc, 'k-', label='val top5 accuracy')
        plt.title('Training and validation top5 accuracy')
        plt.legend()

        plt.show()
        