from preprocessing import ImageTransformer
from random import sample
import os
from shutil import rmtree

import numpy as np
import cv2 as cv


class datasetAugmentationClass:

    def __init__(self, X_in, y_in, transformations, augmented_dataset_size, dataset_dir):
        self.X_in, self.y_in = X_in, y_in
        self.operations = transformations
        m = X_in.shape[0]
        N = int(augmented_dataset_size * m)
        if N == 0:
            N = 1
        X_out_shape = (N,*X_in.shape[1:])
        self.X_out = np.zeros((X_out_shape),dtype='uint8')
        self.y_out = np.zeros((N,9),dtype=int)
        self.dataset_dir = dataset_dir

    def transform_images(self):

        m = self.X_in.shape[0]
        N = self.X_out.shape[0]
        sampling_distribution = sample(range(m),N)
        for i, idx in enumerate(sampling_distribution):
            img_transformer = ImageTransformer(self.operations)
            self.X_out[i,:,:,:] = img_transformer.launch_transform_operation(self.X_in[idx,:,:,:])
            label = np.argmax(self.y_in[idx])
            if 'flip' in self.operations.keys():
                if 'vertical' in self.operations['flip']:
                    flip_mapping = np.array([[0,1,2,3,4,5,6,7,8],[2,1,0,5,4,3,8,7,6]])
                elif 'horizontal' in self.operations['flip']:
                    flip_mapping = np.array([[0,1,2,3,4,5,6,7,8],[6,7,8,3,4,5,0,1,2]])
                new_label = flip_mapping[1,label]
            else:
                new_label = label
            self.y_out[i] = np.zeros((9,),dtype=int)
            self.y_out[i][new_label] = 1
            print()

    def export_augmented_dataset(self):

        # Check if directory exists; if so, delete it. Then create a new one
        dataset_dir = os.path.basename(self.dataset_dir)
        for transformation, value in self.operations.items():
            dataset_dir = dataset_dir + '_' + transformation + '_' + str(value[0])
        dataset_dir = os.path.join(os.path.dirname(self.dataset_dir),dataset_dir)
        if os.path.exists(dataset_dir):
            rmtree(dataset_dir)
        os.mkdir(dataset_dir)

        image_basename = 'transformed_image'
        for transformation, value in self.operations.items():
            image_basename = image_basename + '_' + transformation + '_' + str(value[0])

        m = self.X_out.shape[0]
        labels = np.zeros((m,),dtype=int)
        samples = [[]]*m
        for i,image in enumerate(self.X_out):
            label = np.argmax(self.y_out[i]) + 1
            image_filename = '{}_{}_y={}.jpg'.format(image_basename,(i+1),label)
            filename = os.path.join(dataset_dir,image_filename)
            cv.imwrite(filename,image)

            # Save label data
            labels[i] = label
            samples[i] = image_filename

        # Export label data
        f = open(os.path.join(dataset_dir,'labels.dat'),'w')
        for i in range(m):
            f.write('{},{},\n'.format(samples[i],labels[i]))
        f.close()
