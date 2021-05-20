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


class ModelRadar: #def __init__():
        
           
    @staticmethod
    def show(modelname='cheno',base_dir = 'I:/GHOLAMREZA_Z/ACHENY_LAST_64',fold=1):
        if modelname=='cheno':
            input_shape = (64, 64)
            test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True) 
        else:
            input_shape = (224, 224)
            test_datagen = ImageDataGenerator(rescale=1./255)
            base_dir=base_dir.replace("64","224")
        
        batch_size = 32
        train_dir = base_dir+'/train'
        valid_dir = base_dir+'/validation'
        test_dir = base_dir+'/test'
        CLASS_NO = len(os.listdir(train_dir))
        model=load_model(base_dir+'/'+modelname+'_'+str(fold)+'.h5')    
        num_vars = CLASS_NO
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
        for eval_dir in [test_dir,valid_dir,train_dir] :
            print('\n',modelname,' Evaluate ', eval_dir,' fold ', fold)
            test_generator = test_datagen.flow_from_directory(eval_dir, target_size=input_shape,
                        batch_size=batch_size,class_mode='categorical',shuffle=False)


            # Get the filenames from the generator
            fnames = test_generator.filenames

            # Get the ground truth from generator
            ground_truth = test_generator.classes

            # Get the label to class mapping from the generator
            label2index = test_generator.class_indices
            #print(label2index)
            # Getting the mapping from class index to class label
            idx2label = dict((v,k) for k,v in label2index.items())
            #print(idx2label)
            # Get the predictions from the model using the generator
            predictions = model.predict_generator(test_generator,verbose=1)#, steps=
                                      #validation_generator.samples//validation_generator.batch_size)
            predicted_classes = np.argmax(predictions,axis=1)

            errors = np.where(predicted_classes != ground_truth)[0]
            #print("No of errors = {}/{}".format(len(errors),test_generator.samples))

            class_name = [*test_generator.class_indices.keys()]
            class_all=np.zeros(len(class_name)) 
            class_error=np.zeros(len(class_name)) 
            j=0
            for i in range(test_generator.samples):
                class_all[ground_truth[i]]+=1
                if ground_truth[i] != predicted_classes[i]:
                    class_error[ground_truth[i]]+=1
            #print('{}. {} class:{} predict:{}'.format(j,fnames[i].split('\\')[1], ground_truth[i],predicted_classes[i]))
            #original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
            #plt.figure(figsize=[7,7])
            #plt.axis('off')
            #plt.title('{}. {} class:{} predict:{}'.format(j,fnames[i].split('\\')[1], ground_truth[i],predicted_classes[i]))
            #plt.imshow(original)
            #plt.show()
            #j=j+1
            #if (j==20):
                #break 
                j+=1
            ###print("Total Error : %4d/%4d"%(j,i),end='\r')
            j=0
            values=[]
            for cls in class_name:
                ###print("\n%s (%.3f) %d/%d "%(cls,1-class_error[j]/class_all[j],class_error[j],class_all[j]),end='\r')
                values.append(1-class_error[j]/class_all[j])
                j=j+1
    
    
            values += values[:1]
            if (eval_dir== test_dir): 
                ax.plot(angles, values, color='b', linewidth=1, marker='o', mfc='y', mec='b', 
                        linestyle='dashed', markersize=10,label='Test')
                ax.fill(angles, values, color='g', alpha=0.25)
            elif (eval_dir==valid_dir):
                ax.plot(angles, values, color='b', linewidth=1, marker='s', mfc='w', mec='b', 
                        linestyle='dashed', markersize=9,label='Validation')
                ax.fill(angles, values, color='r', alpha=0.25)
            else:
                ax.plot(angles, values, color='b', linewidth=1, marker='^', mfc='w', mec='b', 
                        linestyle='dashed', markersize=8,label='Train')
                ax.fill(angles, values, color='m', alpha=0.25)
    
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), class_name)
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

            # Ensure radar goes from 0 to 100.
            ##ax.set_ylim(0, 100)
            # You can also set gridlines manually like this:
            #ax.set_rgrids([20, 40, 60, 80, 100])

            # Set position of y-labels (0-100) to be in the middle
            # of the first two axes.
        ax.set_rlabel_position(180 / num_vars)

            # Add some custom styling.
            # Change the color of the tick labels.
        ax.tick_params(colors='k')
            # Make the y-axis (0-100) labels smaller.
        ax.tick_params(axis='y', labelsize=10)
            # Change the color of the circular gridlines.
        ax.grid(color='#606090',linestyle='--')
            # Change the color of the outermost gridline (the spine).
        ax.spines['polar'].set_color('#c0c0ff')
            # Change the background color inside the circle itself.
        ax.set_facecolor('#f0f0f0')

            # Add title.
        ax.set_title(modelname+' Fold'+str(fold)+' Evaluation', y=1.08)

            # Add a legend as well.
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(base_dir+'/'+modelname+'_'+str(fold)+'_Radar', ext="jpg", \
                    close=False, verbose=True, dpi=300, bbox_inches='tight', pad_inches=0.01)
        