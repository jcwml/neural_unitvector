# github.com/jcwml
import sys
import os
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from random import seed, uniform
from time import time_ns
from sys import exit
from os.path import isfile
from os import mkdir
from os.path import isdir

# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices())
# exit();

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# def shuffle_in_unison(a, b):
#     rng_state = np.random.get_state()
#     np.random.shuffle(a)
#     np.random.set_state(rng_state)
#     np.random.shuffle(b)

def norm(v):
    l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return [v[0]/l, v[1]/l, v[2]/l]

def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# hyperparameters
seed(74035)
project = "neural_unitvector"
model_name = 'keras_model'
optimiser = 'adam'
activator = 'tanh'
inputsize = 3
outputsize = 3
epoches = 6
layers = 0
layer_units = 16
batches = 128
samples = 3333333
external_data = 1

# load options
argc = len(sys.argv)
if argc >= 2:
    layers = int(sys.argv[1])
    print("layers:", layers)
if argc >= 3:
    layer_units = int(sys.argv[2])
    print("layer_units:", layer_units)
if argc >= 4:
    batches = int(sys.argv[3])
    print("batches:", batches)
if argc >= 5:
    activator = sys.argv[4]
    print("activator:", activator)
if argc >= 6:
    optimiser = sys.argv[5]
    print("optimiser:", optimiser)
if argc >= 7 and sys.argv[6] == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU_ONLY: 1")
if argc >= 8:
    samples = int(sys.argv[7])
    print("samples:", samples)
if argc >= 9:
    epoches = int(sys.argv[8])
    print("epoches:", epoches)

# make sure save dir exists
if not isdir('models'): mkdir('models')
model_name = 'models/' + activator + '_' + optimiser + '_' + str(layers) + '_' + str(layer_units) + '_' + str(batches) + '_' + str(samples) + '_' + str(epoches)

##########################################
#   CREATE DATASET
##########################################
print("\n--Creating Dataset")
st = time_ns()

if external_data == 0:
    #this dataset is too low quality for some reason ?!
    train_x = np.empty([samples, 3], float)
    train_y = np.empty([samples, 3], float)
    sp = 1.0 / float(samples)
    for i in range(samples):
        m = sp * float(i)
        vec = [uniform(-1,1)*10000, uniform(-1,1)*10000, uniform(-1,1)*10000]
        train_x[i] = vec
        train_y[i] = norm(vec)
else:
    # this dataset produces higher quality training
    if isfile("train_x.npy"):
        train_x = np.load("train_x.npy")
        train_y = np.load("train_y.npy")
    else:
        load_x = []
        with open("dataset.dat", 'rb') as f:
            load_x = np.fromfile(f, dtype=np.float32)
        train_x = np.reshape(load_x, [samples, inputsize])

        train_y = np.empty([samples, 3], float)
        for i in range(samples):
            train_y[i] = norm(train_x[i])
        
        np.save("train_x.npy", train_x)
        np.save("train_y.npy", train_y)

# shuffle_in_unison(train_x, train_y) 
# train_x = np.reshape(train_x, [samples, inputsize])
# train_y = np.reshape(train_y, [samples, outputsize])

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################
print("\n--Training Model")

# construct neural network
model = Sequential()

model.add(Dense(layer_units, activation=activator, input_dim=inputsize))

for x in range(layers):
    # model.add(Dropout(.3))
    model.add(Dense(layer_units, activation=activator))

# model.add(Dropout(.3))
model.add(Dense(outputsize))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'sgd':
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*samples, decay_rate=0.1)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*samples, decay_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False)
    #optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)

model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

# train network
history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches)
model_name = model_name + "_" + "a{:.2f}".format(history.history['accuracy'][-1])
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################
print("\n--Exporting Model")
st = time_ns()

# save weights for C array
print("\nExporting weights...")
li = 0
f = open(model_name + "_layers.h", "w")
f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n")
if f:
    for layer in model.layers:
        total_layer_weights = layer.get_weights()[0].flatten().shape[0]
        total_layer_units = layer.units
        layer_weights_per_unit = total_layer_weights / total_layer_units
        #print(layer.get_weights()[0].flatten().shape)
        #print(layer.units)
        print("+ Layer:", li)
        print("Total layer weights:", total_layer_weights)
        print("Total layer units:", total_layer_units)
        print("Weights per unit:", int(layer_weights_per_unit))

        f.write("const float " + project + "_layer" + str(li) + "[] = {")
        isfirst = 0
        wc = 0
        bc = 0
        if layer.get_weights() != []:
            for weight in layer.get_weights()[0].flatten():
                wc += 1
                if isfirst == 0:
                    f.write(str(weight))
                    isfirst = 1
                else:
                    f.write("," + str(weight))
                if wc == layer_weights_per_unit:
                    f.write(", /* bias */ " + str(layer.get_weights()[1].flatten()[bc]))
                    #print("bias", str(layer.get_weights()[1].flatten()[bc]))
                    wc = 0
                    bc += 1
        f.write("};\n\n")
        li += 1
f.write("#endif\n")
f.close()

# save prediction model
seed(457895)

if external_data == 0:
    predict_samples = 8192
    predict_x = np.empty([predict_samples, 3], float)
    for i in range(predict_samples):
        predict_x[i] = [uniform(-1,1)*10000000, uniform(-1,1)*10000000, uniform(-1,1)*10000000]
else:
    predict_samples = samples
    predict_x = []
    if isfile("predict_x.npy"):
        predict_x = np.load("predict_x.npy")
    else:
        load_x = []
        with open("testset.dat", 'rb') as f:
            load_x = np.fromfile(f, dtype=np.float32)
        predict_x = np.reshape(load_x, [predict_samples, inputsize])
        np.save("predict_x.npy", predict_x)

ad = 0.0
f = open(model_name + "_pd.csv", "w")
if f:
    
    f.write("total deviance | predicted | actual\n")

    p = model.predict(predict_x)
    for i in range(predict_samples):
        an = norm([predict_x[i][0], predict_x[i][1], predict_x[i][2]])
        # dev = abs(p[i][0]-an[0]) + abs(p[i][1]-an[1]) + abs(p[i][2]-an[2])
        dev = dist(p[i][0], p[i][1], p[i][2], an[0], an[1], an[2])
        ad += dev
        f.write(str(dev) + " | " + str(p[i][0]) + "," + str(p[i][1]) + "," + str(p[i][2]) + " | " + str(an[0]) + "," + str(an[1]) + "," + str(an[2]) + "\n")

    f.close()

# save keras model
model.save(model_name)

print("\nTest Set Avg Deviance:", ad/predict_samples, "\n")

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds\n")
