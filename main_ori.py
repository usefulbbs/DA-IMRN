
import os
import numpy as np
from model_SV import unet
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
#from model3d import unet
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam,SGD,Nadam
from keras import backend as K
from keras.layers import concatenate
from keras.models import load_model
from sklearn.metrics import classification_report
import pandas as pd
import datetime
import gc
#tf.enable_eager_execution()
from sklearn.metrics import cohen_kappa_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data_name= "PaviaU"
# data = "PU"
# data_name = "Indian_pines"
# data = "IP"
data_name = "Salinas_valley"
data = "SV"
#data_name= "KSC"
# data_name= "huston"
# data = "HU"
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)
if data_name == "PaviaU": 
    block_size = 10
    patch_size = 8
    classnum = 9
if data_name == "Indian_pines": 
    block_size = 6
    patch_size = 4
    classnum = 16
if data_name == "Salinas_valley": 
    block_size = 10
    patch_size = 8
    classnum = 16
# if data_name == "huston":
#     block_size = 7
#     patch_size = 6
#     classnum = 15


def focal_loss_fixed(y_true, y_pred):
    gamma = 2
    alpha = 0.5
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)    
    model_out = tf.add(y_pred, epsilon)     
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)    
    return tf.reduce_mean(reduced_fl)

def count(gt):
    num_train =[0 for i in range(classnum)]
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for x in range(gt.shape[2]):
                if 1 in gt[i,j,x,0,:]:
                    c = np.argmax(gt[i,j,x,0,:])
                    num_train[c]=num_train[c] +1
    return  num_train
      
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

     
def scheduler(epoch):
    if epoch % 20== 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

    
def reports(X_test,y_test,o):
    pre_list=[0 for i in range(classnum)]
    all_list=[1 for i in range(classnum)]
    AA_list=[0 for i in range(classnum)]
    Y_pred = np.array(model.predict(X_test)) 
    correctnum = 0
    nozero = 0
    arg = []
    test = []
    for batch0 in range(Y_pred.shape[0]):
        for h0 in range(Y_pred.shape[1]):
            for w0 in range(Y_pred.shape[2]):
                if 1 in y_test[batch0,h0,w0,: ,:]:                        
                    i = np.argmax(Y_pred[batch0,h0,w0,0,:])                            
                    j = np.argmax(y_test[batch0,h0,w0,0,:])                
                    nozero = nozero + 1                        
                    all_list[j]=all_list[j]+1
                    if i == j :
                        correctnum = correctnum+1
                        pre_list[i]=pre_list[i]+1
                    arg.append(i)
                    test.append(j)
    kappa_value = cohen_kappa_score(test, arg)
    print("kappa is %f" % round(kappa_value*100, 2))
    totalavg = correctnum / nozero
    print('correct_num : ',correctnum,"  pixel_num :",nozero)
    write_list=[]

    for i in range(0,classnum):
       AA_list[i]= pre_list[i]/all_list[i]
       write_list.append(int(AA_list[i]*10000)/100)
    AA=np.sum(np.array(AA_list)) / classnum
    write_list.append("  ")
    write_list.append(int(totalavg*10000)/100)
    write_list.append(int(AA*10000)/100)
    return totalavg,AA,write_list

def result_print(x,y,name):
    totalavg, AA,write_list= reports(x,y,0) 
    print("_________"+ str(name)+"___________")
    print(str(data_name),"  OA: ",totalavg,"  AA: ",AA)
    return write_list        

for dat in range(1,6):
    for xht in range(0,1) :  
        
        model_num = data+str(dat)+"_"+str(xht)
        X_train = np.load("./predata/"+data+'_'+str(dat)+"/x0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
        train_label = np.load("./predata/"+data+'_'+str(dat)+"/y0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
        X_test = np.load("./predata/"+data+'_'+str(dat)+"/x0testwindowsize" + str(patch_size) + str(data_name)+  ".npy")
        test_label = np.load("./predata/"+data+'_'+str(dat)+"/y0testwindowsize" + str(patch_size) + str(data_name) + ".npy")
        opX_train = np.load("./predata/"+data+'_'+str(dat)+"/opx0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
        optrain_label = np.load("./predata/"+data+'_'+str(dat)+"/opy0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
        X_val = np.load("./predata/"+data+'_'+str(dat)+"/x0valwindowsize" + str(patch_size) + str(data_name)+  ".npy")
        val_label = np.load("./predata/"+data+'_'+str(dat)+"/y0valwindowsize" + str(patch_size) + str(data_name) + ".npy") 
      
        
        X_train = X_train[ :,:,:,:,np.newaxis]    
        train_label = train_label[ :,:,:,np.newaxis]
        opX_train = opX_train[ :,:,:,:,np.newaxis]      
        optrain_label = optrain_label[ :,:,:,np.newaxis]    
        X_val = X_val[ :,:,:,:,np.newaxis]
        val_label = val_label[ :,:,:,np.newaxis]
        X_test = X_test[ :,:,:,:,np.newaxis]
        test_label = test_label[ :,:,:,np.newaxis]
    
        model = unet()    
    #    model = load_model('./models/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5',custom_objects={'focal_loss_fixed':focal_loss_fixed})
       #
       #single loss
        model.compile(optimizer=Nadam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
                      loss=focal_loss_fixed, metrics=['categorical_accuracy'])
        # model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        #               loss=focal_loss_fixed, metrics=['categorical_accuracy'])
        model_checkpoint = ModelCheckpoint('./models/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5',
                                           monitor='val_categorical_accuracy',verbose=1,save_best_only= True)
        earlystopping=EarlyStopping(monitor='val_categorical_accuracy', patience=15, verbose=0, mode='auto')
        reduce_lr = LearningRateScheduler(scheduler)
        callback_lists = [model_checkpoint,earlystopping,reduce_lr]
        history = model.fit(X_train, train_label, batch_size=16, nb_epoch=100,
                 verbose=2, shuffle=True, validation_data=(X_val, val_label),callbacks=callback_lists)
      
        # show current path
#        print("Train_label",count(train_label))
#        print("Val_label",count(val_label))   
#        print("Test_label",count(test_label),sum(count(test_label))) 
        PATH = os.getcwd()
        print (PATH,'./models/'+model_num+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5')
        model = load_model('./models/'+model_num+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5',
                           custom_objects={'focal_loss_fixed':focal_loss_fixed})
        print('.......waiting.........')  
        
        write_list_test = result_print(X_test,test_label,"Test")
        write_list_op = result_print(opX_train,optrain_label,"OP_Train")
#        write_list_val = result_print(X_val,val_label,"Val")
#        write_list_train = result_print(X_train,train_label,"Train")  
#        print(write_list_test)
    
        import os
        import csv
        
        with open('./record/'+ data+"_"+"3ss_result_"+data_name+'.csv', 'a+', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            writer.writerow(write_list_test)
        with open('./record/'+ "OP_"+data+"_"+"3ss_result_"+data_name+'.csv', 'a+', newline='') as csvfileop:
            writerop  = csv.writer(csvfileop)
            writerop.writerow(write_list_op)
            
        del X_train,train_label,X_test,test_label,opX_train,optrain_label,X_val,val_label
        gc.collect()

        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(history.history['categorical_accuracy'], label='training acc')
        plt.plot(history.history['val_categorical_accuracy'], label='val acc')
        plt.title('Accuracy')
        plt.ylabel('Accuracy(%)')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        fig.savefig('./record/' + str(model_num) + str(data_name) + str(block_size) + '-' + str(patch_size) + 'acc.png')
        fig = plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        fig.savefig(
            './record/' + str(model_num) + str(data_name) + str(block_size) + '-' + str(patch_size) + 'loss.png')
        fig = plt.figure()
    #loss accuracy image





















