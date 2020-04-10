import cv2
import numpy as np

class DataAugmentation():
    def __init__(self, im_array):
        if im_array.shape[-1] == 1:
            self.im_array = np.reshape(im_array, im_array.shape[:-1])
            self.augment()
            self.aug_array = np.reshape(self.aug_array, self.aug_array.shape + (1,))
        else: 
            self.im_array = im_array
            self.augment()
        

    def flip(self, im_array, vertical=True, horizontal=True):
        if vertical and horizontal:
            flipcode = -1
        elif horizontal:
            flipcode = 1
        elif vertical:
            flipcode = 0
        else: flipcode=None
        
        if not(flipcode):
            print('no flip, return original array.')
            return im_array
        else:
            return cv2.flip(im_array, flipCode=flipcode)
        
        
    def rot180(self, im_array):

        return cv2.rotate(im_array, cv2.ROTATE_180)    


    def center_crop_scale(self, im_array, ratio=0.5):
        # crop an image out from center
        width, height = im_array.shape[0], im_array.shape[1]
        center_x, center_y = np.int(np.floor(width/2)), np.int(np.floor(height/2)) 
        dx, dy = np.int(width*ratio/2), np.int(height*ratio/2)
        new_array = np.array(im_array[center_x-dx:center_x+dx, center_y-dy:center_y+dy], dtype='uint8')
        # scale to original shape
        return cv2.resize(new_array, im_array.shape)


    def augment(self):
        # create separate sets of augmented data using flip, rotate and crop methods
        im_array = self.im_array
        im_flip = np.array([self.flip(im_array[n_im, :, :]) for n_im in range(im_array.shape[0])])        
        im_rot180 = np.array([self.rot180(im_array[n_im, :, :]) for n_im in range(im_array.shape[0])])        
        im_crop = np.array([self.center_crop_scale(im_array[n_im, :, :]) for n_im in range(im_array.shape[0])])

        self.aug_array = np.concatenate((im_array, im_flip, im_rot180, im_crop), axis=0)       
 

               



