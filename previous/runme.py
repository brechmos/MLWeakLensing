import os 
from astropy.io import fits
import numpy as np
import random

from keras.models import Sequential
from keras.utils import np_utils

# convlulation layres to help train on image data
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D

# optimizers
from keras.optimizers import SGD

# core layers
from keras.layers import Activation, Dropout, Flatten, Dense

# This is to mimic the conv_WL_devel file.

display = True
labels = ['750', '850']
path='/Users/crjones/Documents/Science/HargisDDRF/astroNN/data/wl_maps'
degrade=8
nct = 9
imsize=32

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# --------------------------------------------------------------------------
# Read in the data
# --------------------------------------------------------------------------
if True:
    imgs = np.zeros([2048//degrade, 2048//degrade, nct, len(labels)])
    imgs2 = {'750': [], '850': []} 
    for j, label in enumerate(labels):
        for i in range(nct):
            filename = os.path.join(path, 'smoothWL-conv_m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.'+label+'_4096xy_000'+ np.str(i+1) +'r_0029p_0100z_og.gre.fit')
            if display: 
               print("i: %d  j: %d  name: %s" % (i, j, 'smoothWL-conv_m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.'+label+'_4096xy_000'+ np.str(i+1) +'r_0029p_0100z_og.gre.fit'))

            # Read in the data and put into the imgs.
            f = fits.open(filename)
            imgs[:,:,i,j]=rebin(f[0].data, [2048//degrade, 2048//degrade])

    #  Create the 8x8 sub images from the image just read in
    subimgs = {'750': [], '850': []} 
    for j, label in enumerate(labels):
        for i in range(nct):

            # Grab the list of 8x8 sub arrays
            temp_subimgs = [x for x in blockshaped(imgs[:,:,i,j], imsize, imsize)]

            # Now create all the flips, rotations, transpositions etc
            subimgs[label].extend(temp_subimgs)  # sub images
            subimgs[label].extend([x.T for x in temp_subimgs]) # transposed sub images
            subimgs[label].extend([np.rot90(x) for x in temp_subimgs]) # rotated sub images
            subimgs[label].extend([np.rot90(x, k=2) for x in temp_subimgs]) # rotated twice sub images
            subimgs[label].extend([np.rot90(x, k=3) for x in temp_subimgs]) # rotated three x sub images
            subimgs[label].extend([x[::-1] for x in temp_subimgs]) # flip ud
            subimgs[label].extend([x[:,::-1] for x in temp_subimgs]) # flip lr


# --------------------------------------------------------------------------
# Create training data
# --------------------------------------------------------------------------
N_training = int(0.9 * len(subimgs[label]) * 2)

# Create the X_train and y_train from the '750' data
training_random_inds_750 = np.random.choice(range(len(subimgs['750'])), N_training//2)
X_train = [subimgs['750'][x]  for x in training_random_inds_750]
y_train = [750]*(N_training//2)

# Extend the X_train and y_train with the '850' data
training_random_inds_850 = np.random.choice(range(len(subimgs['850'])), N_training//2)
X_train.extend([subimgs['850'][x]  for x in training_random_inds_850])
y_train.extend([850]*(N_training//2))

# Randomize the list of training data
inds = [x for x in range(len(X_train))]
random.shuffle(inds)
X_train = [X_train[ii] for ii in inds]
y_train = [y_train[ii] for ii in inds]

# Now make them into a 3D array and 1D array
X_train = np.stack(X_train)
y_train = np.array(y_train)

# Add dimension to X_train for keras
X_train = X_train.reshape(X_train.shape[0], 1, imsize, imsize)

# change y_train to labels of 0 and 1 (conversion of boolean to int)
y_train = np.array(y_train == 850)*1
y_train = np_utils.to_categorical(y_train, 2)

# YIKES https://github.com/ml4a/ml4a-guides/issues/10  !!!!!!!!!!!
from keras import backend as K
K.set_image_dim_ordering('th')

# --------------------------------------------------------------------------
# Next setup Keras
# --------------------------------------------------------------------------
model = Sequential()

# input is 4D tensor with shape: (samples, channels, rows, cols)
# output is 4D tensor with shape: (samples, filters, new_rows, new_cols)
model.add(Conv2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, imsize, imsize)))

model.add(Conv2D(64, (5, 5), strides=(1,1), activation='relu'))

#model.add(Conv2D(64, (1, 1), activation='relu'))

# MaxPooling2D is a way to reduce the number of parameters in our model by 
# sliding a 2x2 pooling filter across the previous layer and taking the max of 
# the 4 values in the 2x2 filter.
#model.add(MaxPooling2D(pool_size=(1,1)))

# https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
        optimizer='rmsprop', metrics=['accuracy', 'mae', 'mape'])

# --------------------------------------------------------------------------
# Fit model on training data
# --------------------------------------------------------------------------

model.fit(X_train, y_train, nb_epoch=20, verbose=1)

# --------------------------------------------------------------------------
# Create testing data
# --------------------------------------------------------------------------

# Create test inds for 750
possible_test_inds_750 = list(set(range(len(subimgs['750']))) - set(training_random_inds_750))

N_testing = max(3000, int(0.2*len(possible_test_inds_750)))

test_random_inds_750 = np.random.choice(possible_test_inds_750, N_testing//2)
X_test = [subimgs['750'][x]  for x in test_random_inds_750]
y_test = [750]*(N_testing//2)

# Extend the X_test and y_test with the '850' data
possible_test_inds_850 = list(set(range(len(subimgs['850']))) - set(training_random_inds_850))
test_random_inds_850 = np.random.choice(possible_test_inds_850, N_testing//2)
X_test.extend([subimgs['850'][x]  for x in test_random_inds_850])
y_test.extend([850]*(N_testing//2))

# Randomize the list of training data
inds = [x for x in range(len(X_test))]
random.shuffle(inds)
X_test = [X_test[ii] for ii in inds]
y_test = [y_test[ii] for ii in inds]

# Now make them into a 3D array and 1D array
X_test = np.stack(X_test)
y_test = np.array(y_test)

# Add dimension to X_train for keras
X_test = X_test.reshape(X_test.shape[0], 1, imsize, imsize)

# change y_test to labels of 0 and 1 (conversion of boolean to int)
y_test = np.array(y_test == 850)*1
y_test = np_utils.to_categorical(y_test, 2)

# --------------------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------------------
print('Evaluating model...')

scores = model.evaluate(X_test, y_test, verbose=1)
names = model.metrics_names
print('{}={}, {}={}, {}={}, {}={}'.format(names[0], scores[0], names[1], scores[1], names[2], scores[2], names[3], scores[3]))
