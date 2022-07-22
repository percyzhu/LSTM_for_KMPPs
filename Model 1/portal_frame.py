# Setup GPU for training (use tensorflow v2.7)
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use '-1' for CPU only, use '0' if there is only 1 GPU
                                          # use GPU number for multiple GPUs
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.io
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.compat.v1.keras.layers import LSTM, Activation, CuDNNLSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time
from random import shuffle
import joblib  # save scaler

# fonts
font = {'family':'Times New Roman'}
matplotlib.rc('font',**font)
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Times New Roman']  

# Load data
dataDir = 'D:/new/'  # Replace the directory
mat = scipy.io.loadmat(dataDir+'data.mat')

X_data = mat['input_tf']
y_data = mat['target_tf']
train_indices = mat['trainInd'] - 1
test_indices = mat['valInd'] - 1

# Scale data
X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], X_data.shape[2]])
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_X.fit(X_data_flatten)
X_data_flatten_map = scaler_X.transform(X_data_flatten)
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], X_data.shape[2]])

y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], y_data.shape[2]])
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y.fit(y_data_flatten)
y_data_flatten_map = scaler_y.transform(y_data_flatten)
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

# Unknown data
X_pred = mat['input_pred_tf']
y_pred_ref = mat['target_pred_tf']

# Scale data
X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])

y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

X_data_new = X_data_map
y_data_new = y_data_map

X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

X_pred = X_pred_map
y_pred_ref = y_pred_ref_map

data_dim = X_train.shape[2]  # number of input features
timesteps = X_train.shape[1]
num_classes = y_train.shape[2]  # number of output features
batch_size = 128

# Delete variables not in use
del X_data_flatten,X_data_flatten_map,y_data_flatten,y_data_flatten_map,X_data_map
del y_data_map,X_pred_flatten,X_pred_flatten_map,X_pred_map,y_pred_ref_flatten,y_pred_ref_flatten_map
del y_pred_ref_map

# Neural Network model
rms = RMSprop(learning_rate=0.001, decay=0.0001)
adam = Adam(learning_rate=0.001, decay=0.0001)
model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, data_dim)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Activation('relu'))
model.add(Dense(num_classes))
model.summary()
model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_squared_error, mean_absolute_error
              optimizer=adam,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])
best_loss = 100
train_loss = []
test_loss = []
history = []

# Training
with tf.device('/device:GPU:0'):
     start = time.time()

     epochs = 50000
     for e in range(epochs):
         print('epoch = ', e + 1)

         if e >= 1:
             model = load_model(dataDir + 'results/temp_model.h5')
         Ind = list(range(len(X_data_new)))
         shuffle(Ind)
         ratio_split = 0.7
         Ind_train = Ind[0:round(ratio_split * len(X_data_new))]
         Ind_test = Ind[round(ratio_split * len(X_data_new)):]

         X_train = X_data_new[Ind_train]
         y_train = y_data_new[Ind_train]
         X_test = X_data_new[Ind_test]
         y_test = y_data_new[Ind_test]

         model.fit(X_train, y_train,
                   batch_size=batch_size,
                   # validation_split=0.2,
                   validation_data=(X_test, y_test),
                   shuffle=True,
                   use_multiprocessing=True,
                   workers=12,                  
                   epochs=1)
         score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
         score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
         train_loss.append(score0[0])
         test_loss.append(score[0])

         model.save(dataDir + 'results/temp_model.h5')
         if test_loss[e] < best_loss:
             best_loss = test_loss[e]
             model.save(dataDir + 'results/my_best_model.h5')

         del model
         K.clear_session()
         gc.collect()
        
         if (e+50)%100==0:
            time.sleep(100)
       
     end = time.time()
     running_time = (end - start)/3600
     print('Running Time: ', running_time, ' hour')

# Plot history of loss functions
figsize = 9,6
figure, ax = plt.subplots(figsize=figsize)
plt.plot(np.array(train_loss), 'k-',label='Train loss',alpha=0.5)
plt.plot(np.array(test_loss), 'r-',label='Test loss',alpha=0.5)
plt.yscale("log")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.tick_params(axis='both', which='major', tickdir='in', length=5)
ax.tick_params(axis='both', which='minor', tickdir='in', length=3)
plt.xlabel('episode number',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(frameon=False,fontsize=24)
plt.margins(0.05)
plt.subplots_adjust(top=0.95,bottom=0.15,left=0.12,right=0.95,hspace=0,wspace=0)
plt.savefig(r'figures\loss.svg')
plt.close(figure)

# fonts
font = {'family':'HGSS2_CNKI'}
matplotlib.rc('font',**font)
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['HGSS2_CNKI'] 

# Load model for evaluation
model_best = load_model(dataDir + 'results/my_best_model.h5')

# Inputs and outputs
X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

y_train_pred = model_best.predict(X_train)
y_test_pred = model_best.predict(X_test)
y_pure_preds = model_best.predict(X_pred)

# Reverse map to original magnitude
X_train_orig = X_data[0:len(train_indices[0])]
y_train_orig = y_data[0:len(train_indices[0])]
X_test_orig = X_data[len(train_indices[0]):]
y_test_orig = y_data[len(train_indices[0]):]
X_pred_orig = mat['input_pred_tf']
y_pred_ref_orig = mat['target_pred_tf']

# Plot training samples
y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0]*y_train_pred.shape[1], y_train_pred.shape[2]])
y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

x_time = range(0,3610,10)

for i in range(len(y_train)):
    for j in range(y_train.shape[2]):
        figsize = 8,6
        figure, ax = plt.subplots(figsize=figsize)
        plt.plot(x_time,y_train_orig[i][:, j], 'r-', label='True',alpha=0.5)
        plt.plot(x_time,y_train_pred[i][:, j], 'k-', label='Prediction',alpha=0.5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.tick_params(axis='both', which='major', tickdir='in', length=5)
        ax.tick_params(axis='both', which='minor', tickdir='in', length=3)
        plt.xlabel('time / s',fontsize=25)
        plt.ylabel('displacement / mm',fontsize=25)
        plt.legend(frameon=False,fontsize=25)
        plt.margins(0.1)
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.2,right=0.95,hspace=0,wspace=0)
        plt.savefig(r'figures\training_'+'{}_dis_{}.png'.format(i,j),dpi=300)
        plt.close(figure)
   
# Plot testing samples
y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0]*y_test_pred.shape[1], y_test_pred.shape[2]])
y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

for i in range(len(y_test)):
    for j in range(y_test.shape[2]):
        figsize = 8,6
        figure, ax = plt.subplots(figsize=figsize)
        plt.plot(x_time,y_test_orig[i][:, j], 'r-', label='True',alpha=0.5)
        plt.plot(x_time,y_test_pred[i][:, j], 'k-', label='Prediction',alpha=0.5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.tick_params(axis='both', which='major', tickdir='in', length=5)
        ax.tick_params(axis='both', which='minor', tickdir='in', length=3)
        plt.xlabel('time / s',fontsize=25)
        plt.ylabel('displacement / mm',fontsize=25)
        plt.legend(frameon=False,fontsize=25)
        plt.margins(0.1)
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.2,right=0.95,hspace=0,wspace=0)
        plt.savefig(r'figures\testing_'+'{}_dis_{}.png'.format(i,j),dpi=300)
        plt.close(figure)

# Plot pure prediction samples
y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0]*y_pure_preds.shape[1], y_pure_preds.shape[2]])
y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

for i in range(len(y_pred_ref)):
    for j in range(y_pred_ref.shape[2]):
        figsize = 8,6
        figure, ax = plt.subplots(figsize=figsize)
        plt.plot(x_time,y_pred_ref_orig[i][:, j], 'r-', label='True',alpha=0.5)
        plt.plot(x_time,y_pure_preds[i][:, j], 'k-', label='Prediction',alpha=0.5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.tick_params(axis='both', which='major', tickdir='in', length=5)
        ax.tick_params(axis='both', which='minor', tickdir='in', length=3)
        plt.xlabel('time / s',fontsize=25)
        plt.ylabel('displacement / mm',fontsize=25)
        plt.legend(frameon=False,fontsize=25)
        plt.margins(0.1)
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.2,right=0.95,hspace=0,wspace=0)
        plt.savefig(r'figures\pred_'+'{}_dis_{}.png'.format(i,j),dpi=300)
        plt.close(figure)

# Save scaler
joblib.dump(scaler_X, dataDir+'results/scaler_X.save')
joblib.dump(scaler_y, dataDir+'results/scaler_y.save')
# Load scalers
# scaler_X = joblib.load(dataDir+'results/scaler_X.save')
# scaler_y = joblib.load(dataDir+'results/scaler_y.save')

 # Save all the information
scipy.io.savemat(dataDir+'results/results_BW.mat',
                   {'y_train': y_train, 'y_train_orig': y_train_orig, 'y_train_pred': y_train_pred,
                   'y_test': y_test, 'y_test_orig': y_test_orig, 'y_test_pred': y_test_pred,
                   'y_pred_ref': y_pred_ref, 'y_pred_ref_orig': y_pred_ref_orig, 'y_pure_preds': y_pure_preds,
                   'X_train': X_train, 'X_test': X_test, 'X_pred': X_pred,
                   'train_indices': train_indices[0], 'test_indices': test_indices[0], #'pred_indices': pred_indices[0],
                   'train_loss': train_loss, 'test_loss': test_loss, 'best_loss': best_loss,
                   'running_time': running_time, 'epochs': epochs})