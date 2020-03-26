import numpy as np
import os
from PIL import Image
from ImagePreprocess import ImagePreprocess
import cv2
import time
class LoadImage:
    # load files of one image at a time given the pathname
    def __init__(self, pathname, image_type='test'):
        self.pathname = pathname
        self.image_type = image_type        
        self.dict_path_file()


    def find_subfolder(self, pathname):
        # subfolders = [os.path.join(pathname,d) for d in os.listdir(pathname) if os.path.isdir(os.path.join(pathname,d))]
        subfolders = [d for d in os.listdir(pathname) if os.path.isdir(os.path.join(pathname,d))]

        return subfolders        
        
    def find_file(self, pathname):
        # files = [os.path.join(pathname,d) for d in os.listdir(pathname) if os.path.isfile(os.path.join(pathname,d))]
        files = [d for d in os.listdir(pathname) if os.path.isfile(os.path.join(pathname,d))]
        return files

    def dict_path_file(self):
        print('reading file locations')
        # find image folder
        im_list = self.find_subfolder(self.pathname)
        # find the subfolders under each image folder
        dict_files = {}
        for n in range(len(im_list)):
            d = {'masks':[]}
            im_name = im_list[n]
            n_mask = None
            if self.image_type == 'train':
                # training data contains two folders under each item ../images, ../           
                list_mask = self.find_file(os.path.join(self.pathname, im_list[n] + '/masks/'))
                n_mask = len(list_mask)
                d['masks'] = list_mask
            
            dict_files[im_name] = d   
        self.dict_files = dict_files
        
    def load_image(self,filepath):
        im = Image.open(filepath).convert('L')
        im_array = np.asarray(im)
        return im_array
    
    def pile_image_matrix(self):
        # load all training images by filepathss         
        # preprocess: resize images and normalize pixel values
        # relabel image matrix
        # overlap image matrix
        # save processed image matrix to .npy files
        list_train = np.array(list(self.dict_files.keys()))
        n_img = len(list_train)
        shape = [256, 256]
        img_matrix = np.empty((n_img, shape[0], shape[1]))  
        mask_matrix = np.empty((n_img, shape[0], shape[1]))
        for i, it in enumerate(list_train):
            t_start = time.time()
            imgpath = self.pathname + '/' + it + '/images/' + it + '.png'
            m = self.load_image(imgpath)
            m = cv2.resize(m, (shape[0], shape[1]))
            img_matrix[i, :, :] = m
            if self.image_type == 'train':
                maskpath = [self.pathname + '/' + it + '/masks/' + m for m in self.dict_files[it]['masks']]
                mask = np.empty((len(maskpath), shape[0], shape[1]))
                for imp, mp in enumerate(maskpath):
                    m = self.load_image(mp)
                    m = cv2.resize(m, (shape[0], shape[1]))
                    mask[imp, :, :] = m
                I_mask = ImagePreprocess(mask, datatype='masks')
                mask_matrix[i, :, :] = I_mask.mask_matrix
            t_spent = time.time() - t_start
            print('image', str(i), ',time=', str(t_spent),'s')
        I_image = ImagePreprocess(img_matrix, datatype='images')
        image_matrix = I_image.image_norm

        return image_matrix, mask_matrix
