import os
import numpy as np
import scipy.misc
import h5py
#np.random.seed(123)
from imgaug import augmenters as iaa
from PIL import Image
import cv2

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])
        self.lab_set = np.array(f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        self.shuffle()
        self._idx = 0
        
    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        
        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                # Rotation
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    angle = np.random.random_integers(0,45)
                    M = cv2.getRotationMatrix2D((self.load_size/2,self.load_size/2),angle,1)
                    image = cv2.warpAffine(image,M,(self.load_size,self.load_size))
                flip = np.random.random_integers(0, 1)
                # Flip Left Right
                if flip>0:
                    image = image[:,::-1,:]
                #Gaussian Blur
                flip = np.random.random_integers(0,1)
                if flip >0:
                    kernel_size = np.random.random_integers(1,5)
                    sigma = np.random.random_integers(0,3)
                    image = cv2.GaussianBlur(image,(kernel_size*2+1,kernel_size*2+1),sigma)
                #Pixel Dropout
                flip = np.random.random_integers(0,1)
                if flip >0:
                    pixel_numbers = np.random.random_integers(100,200)
                    for repeat in range(pixel_numbers):
                        x_pixel = np.random.random_integers(4,self.load_size-4)
                        y_pixel = np.random.random_integers(4,self.load_size-4)
                        half_kernel_size = np.random.random_integers(1,3)
                        kernel_size = half_kernel_size*2+1
                        for j in range(kernel_size):
                            for k in range(kernel_size):
                                image[x_pixel-half_kernel_size+j][y_pixel-half_kernel_size+k] = 0.0
                #Gaussian Noise
                flip = np.random.random_integers(0,1)
                if flip >0:
                    sigma = 0.5
                    gaussian_noise = np.random.normal(0,sigma,np.shape(image))
                    image = image + gaussian_noise
                #Color Shift
                flip = np.random.random_integers(0,1)
                if flip > 0:
                    color_shift = np.zeros(np.shape(image))
                    for k in range(3):
                        change = np.random.normal(0,0.1)
                        color_shift[:,:,k] = np.full_like(color_shift[:,:,k],change)
                    image = image+color_shift



                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)



                

            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.randomize:
                    self.shuffle()
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                # Rotation
                flip = np.random.random_integers(0, 1)
                if flip>8:
                    angle = np.random.random_integers(0,45)
                    M = cv2.getRotationMatrix2D((self.load_size/2,self.load_size/2),angle,1)
                    image = cv2.warpAffine(image,M,(self.load_size,self.load_size))
                #Flip Left Right
                flip = np.random.random_integers(0, 1)
                if flip>8:
                    image = image[:,::-1,:]
                #Gaussian Blur
                flip = np.random.random_integers(0,1)
                if flip >0:
                    kernel_size = np.random.random_integers(1,5)
                    sigma = np.random.random_integers(0,3)
                    image = cv2.GaussianBlur(image,(kernel_size*2+1,kernel_size*2+1),sigma)
                #Pixel Dropout
                flip = np.random.random_integers(0,1)
                if flip >0:
                    pixel_numbers = np.random.random_integers(100,200)
                    for repeat in range(pixel_numbers):
                        x_pixel = np.random.random_integers(4,self.load_size-4)
                        y_pixel = np.random.random_integers(4,self.load_size-4)
                        half_kernel_size = np.random.random_integers(1,3)
                        kernel_size = half_kernel_size*2+1
                        for j in range(kernel_size):
                            for k in range(kernel_size):
                                image[x_pixel-half_kernel_size+j][y_pixel-half_kernel_size+k] = 0.0
                #Gaussian Noise
                flip = np.random.random_integers(0,1)
                if flip >0:
                    sigma = 0.5
                    gaussian_noise = np.random.normal(0,sigma,np.shape(image))
                    image = image + gaussian_noise
                #Color Shift
                flip = np.random.random_integers(0,1)
                if flip > 0:
                    color_shift = np.zeros(np.shape(image))
                    for k in range(3):
                        change = np.random.normal(0,0.1)
                        color_shift[:,:,k] = np.full_like(color_shift[:,:,k],change)
                    image = image+color_shift

                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
                


            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

# Loading test data from disk with no labels
class DataLoaderTestDisk(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        self.list_im = []
        self.files = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path = line.strip('\n')
                self.list_im.append(os.path.join(self.data_root, path))
                self.files.append(path)
        self.list_im = np.array(self.list_im, np.object)
        self.num = self.list_im.shape[0]
        print('# Test images found:', self.num)

    def get_test_images(self):
        images = np.zeros((self.num, self.fine_size, self.fine_size, 3))
        for i in xrange(self.num):
            image = scipy.misc.imread(self.list_im[i])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean            
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]

        return images

    def get_file_list(self):
        return self.files

    def size(self):
        return self.num

# Loading test data from H5 with no labels
class DataLoaderTestH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])

        self.num = self.im_set.shape[0]

        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Test images found:', self.num)

        self.shuffle()

    def next_batch(self):
        images_batch = np.zeros((self.num, self.fine_size, self.fine_size, 3)) 
        for i in range(self.num):
            image = self.im_set[i]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            
            # if self._idx == self.num:
            #     self._idx = 0
            #     if self.randomize:
            #         self.shuffle()
        
        return images_batch

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 

    def size(self):
        return self.num
