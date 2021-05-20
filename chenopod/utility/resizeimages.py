from imutils import paths
import numpy as np
import random
import cv2
import os, shutil

class ResizeImages:
    @staticmethod 
    def doresize(new_dimention= (64, 64), destination_dir="I:/GHOLAMREZA_Z/ACHENY_LAST_64"):
        SRC = "I:/GHOLAMREZA_Z/ACHENY_LAST/new_wild"
        VALID=0.20
        TEST=0.10
        TRAIN=1-VALID-TEST

        if not os.path.exists(destination_dir):
            os.mkdir(destination_dir)
        else:
            print(destination_dir,' already exist')
            return
        
        for folder in ['/train','/validation','/test']:
            os.mkdir(destination_dir+folder)
        k=1
        allclass=os.listdir(SRC)
        for cls in allclass:
            for folder in ['/train/','/validation/','/test/']:
                if not os.path.exists(destination_dir+folder+cls):
                    os.mkdir(destination_dir+folder+cls)        
            files = list(paths.list_images(SRC+"/"+cls))
            total=len(files)
            temp=[]
            if total>1000:
                newlen=int(0.9*total)
                delta=total/newlen
                jit=delta
                for kk in range(total):
                    if(kk==int(jit)):
                        temp.append(files[kk])
                        jit+=delta
                files=temp
                total=len(files)
            split_1 = int(TRAIN * total)
            split_2 = int((TRAIN+VALID) * total)
            test_split1=int(TEST * total/2)
            train = files[test_split1:split_1+test_split1]
            valid = files[split_1+test_split1:split_2+test_split1]
            test = files[:test_split1] + files[split_2+test_split1:]
            print("\n\r%d)%s [All:%d] train:%d valid:%d test:%d"%(k,cls,total,len(train),len(valid),len(test)))
            f=1
            for file in train: 
                image = cv2.imread(file)
                image64=cv2.resize(image,new_dimention)       
                cv2.imwrite(destination_dir+'/train/'+cls+'/'+file[file.rfind('\\')+1:],image64)
                print('%4d'%(f),end='\r')
                f=f+1
            for file in valid: 
                image = cv2.imread(file)
                image64=cv2.resize(image,new_dimention)       
                cv2.imwrite(destination_dir+'/validation/'+cls+'/'+file[file.rfind('\\')+1:],image64)
                print('%4d'%(f),end='\r')
                f=f+1
            for file in test: 
                image = cv2.imread(file)
                image64=cv2.resize(image,new_dimention)       
                cv2.imwrite(destination_dir+'/test/'+cls+'/'+file[file.rfind('\\')+1:],image64)
                print('%4d'%(f),end='\r')
                f=f+1
            k=k+1

    @staticmethod
    def showstat(destination_dir="I:/GHOLAMREZA_Z/ACHENY_LAST_64"):
        if not os.path.exists(destination_dir):
            print(destination_dir,' already not exist')
            return
        nt_t=0; nv_t=0; ns_t=0; i=1
        train_dir=destination_dir+'/train';valid_dir=destination_dir+'/validation';test_dir=destination_dir+'/test'
        all_class=os.listdir(train_dir)
        print("{0:2d}  Name    #Total    # Train    # Validation    # Test".format(len(all_class)))
        print("------------------------------------------------------------")
        for c in all_class:
            nt=len(os.listdir(train_dir+'/'+c)); nt_t += nt
            nv=len(os.listdir(valid_dir+'/'+c)); nv_t += nv
            ns=len(os.listdir(test_dir+'/'+c)); ns_t += ns
            print("{0:2d} {1:6} {2:7d} {3:10d} {4:12d} {5:12d}".format(i,c,nt+nv+ns,nt,nv,ns))
            i+=1
        print("------------------------------------------------------------")    
        print("{0:12} {1:5d} {2:10d} {3:12d} {4:13d}".format("Total",nt_t+nv_t+ns_t,nt_t,nv_t,ns_t)) 
        