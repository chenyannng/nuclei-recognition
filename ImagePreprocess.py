import numpy as np
import cv2

class ImagePreprocess:
    def __init__(self, all_matrix, datatype='images'):
        self.all_matrix = all_matrix
        
        if datatype == 'images':            
            self.image_norm = self.zero_mean()

        elif datatype == 'masks':
            masks = np.zeros((all_matrix.shape[1], all_matrix.shape[2]))
            for i in range(all_matrix.shape[0]):
                im = all_matrix[i, :, :]
                im = self.scale(im)
                im = self.find_contour(im)
                # print( np.sum(im[:]) == 0)
                # print(np.isinf(im[:]))
                unique, unique_counts = np.unique(im,  return_counts=True)
                if len(unique) == 1 or np.isinf(im[:]).any():
                    # skip all black images and image with inf values
                    print('skip mask image with invalid values:',str(i))
                else:
                    masks = np.maximum(masks, im)
                
            self.mask_matrix = masks
            
        else: 
            print('datatype= images or masks.')    

    def zero_mean(self):    
    # image data pre-processing, normalize pixel values to zero-mean
    # data is a numpy array of shape (i, n, m), i is number of images, n and m are 2D image dimensions
    # mean subtract,
        mean_all = np.mean(self.all_matrix, axis = 0)
        norm_all = np.empty(self.all_matrix.shape)
        for im in range(self.all_matrix.shape[0]):
            norm_one = self.all_matrix[im,:,:] - mean_all
            norm_all[im,:,:] = norm_one
            
        return norm_all


    def scale(self, im_array):
        # scale image pixel value to [0, 1.]
        im_array = np.divide(im_array, np.max(im_array))
        
        return im_array


    def find_contour(self, im_array):

        im = self.pad(im_array)
        im1 = np.multiply(im[0:-2, 0:-2], im[2:, 2:])
        im2 = np.multiply(im[0:-2, 2:], im[2:, 0:-2])
        im = np.multiply(im1, im2)
        #label pixels by 0=outer, 1=inner, 2=contour
        contour = im_array - im
        im[contour!=0] = 2

        return im

    def pad(self, im_array):
        # pad zero arround image array
        new_array = np.zeros((im_array.shape[0]+2, im_array.shape[1]+2))
        new_array[1:im_array.shape[0]+1, 1:im_array.shape[0]+1] = im_array
        
        return new_array 

    