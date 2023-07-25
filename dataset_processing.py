import os, re
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import ImageTransformer
 

def preprocess_data_tf(im, label):

    im = tf.cast(im, tf.float32)
    im = im/127.5
    im = im - 1

    return im, label

def preprocess_data(im_tilde, label):

    im_tilde = im_tilde.astype(np.float32)
    im_tilde = im_tilde/127.5
    im_tilde = im_tilde - 1

    return im_tilde, label


def set_dataset(case_dir, img_dims, dataset_foldername):

    datasets_dir = os.path.join(case_dir,'Datasets',dataset_foldername)
    x_cont = []
    y_cont = []
    for folder in os.listdir(datasets_dir): # for each dataset contained in the Dataset folder
        f = open(os.path.join(datasets_dir,folder,'labels.dat'))
        data = f.read()
        # Read labels
        labels = [int(label) for label in re.findall('\n*.*(\d)\n*',data)]
        # Read samples
        samples = re.findall('\n*(.*),\d\n*',data)
        samples_path = [os.path.join(datasets_dir,folder,sample) for sample in samples]

        # Generate X,y datasets
        m = len(samples_path)
        x = np.zeros((m,img_dims[1],img_dims[0],3),dtype='uint8')
        y = np.zeros((m,),dtype=int)
        for i,sample in enumerate(samples_path):
            # X-array storage
            img = cv.imread(sample)
            x[i,:,:,:] = ImageTransformer.resize(img,img_dims)
            # Label storage
            y[i] = labels[i]

        x_cont.append(x)
        y_cont.append(y)

    m = [item.shape[0] for item in x_cont]
    X = np.zeros([sum(m),*x.shape[1:]],dtype='uint8')
    Y = np.zeros([sum(m),],dtype=int)
    for i in range(len(m)):
        X[sum(m[0:i]):sum(m[0:i+1]),:,:,:] = x_cont[i]
        Y[sum(m[0:i]):sum(m[0:i+1]),] = y_cont[i]

    return X, Y

def read_preset_datasets(case_dir, dataset_ID=None, return_filepaths=False):

    if dataset_ID == None:
        dataset_dir = [case_dir]
    else:
        dataset_dir = [os.path.join(case_dir,'Dataset_{}'.format(i)) for i in dataset_ID]

    X = []
    y = []
    for folder in dataset_dir:
        if os.path.exists(folder):
            files = [os.path.join(folder,file) for file in os.listdir(folder)]
            for i,file in enumerate(files):
                X.append(cv.imread(file))
                label = re.search('\w+\_\d+\_y\=(\d).*',os.path.basename(file))
                y.append(label)
        else:
            X = None
            y = None
    X = np.array(X,dtype='uint8')
    y = np.array(y,dtype=int)

    if return_filepaths:
        return X, y, files
    else:
        return X, y

def get_test_dataset(case_dir, img_dims):

    # Read original datasets
    X, y = set_dataset(case_dir,img_dims)

    return (X,y)

def get_datasets(case_dir, img_dims, train_size):

    # Read original datasets
    X, y = set_dataset(case_dir,img_dims,dataset_foldername='Training')

    X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=train_size,shuffle=True)
    X_cv, X_test, y_cv, y_test = train_test_split(X_val,y_val,train_size=0.75,shuffle=True)

    data_train = (X_train, y_train)
    data_cv = (X_cv, y_cv)
    data_test = (X_test, y_test)
    
    return data_train, data_cv, data_test

def create_dataset_pipeline(dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    X, y = dataset
    y_oh = tf.one_hot(y,depth=9)
    dataset_tensor = tf.data.Dataset.from_tensor_slices((X, y_oh))

    if is_train:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=X.shape[0]).repeat()
    dataset_tensor = dataset_tensor.map(preprocess_data_tf, num_parallel_calls=num_threads)
    dataset_tensor = dataset_tensor.batch(batch_size)
    dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

    return dataset_tensor

    
def get_tensorflow_datasets(data_train, data_cv, data_test, batch_size=32):

    dataset_train = create_dataset_pipeline(data_train,is_train=True,batch_size=batch_size)
    dataset_cv = create_dataset_pipeline(data_cv,is_train=False,batch_size=1)
    dataset_test = preprocess_data_tf(data_test[0],tf.one_hot(data_test[1],depth=1))
    
    return dataset_train, dataset_cv, dataset_test