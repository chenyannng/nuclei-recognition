import numpy as np
from LoadImage import LoadImage

# load image data, do preprocess and save matrix in .npy files
trainpath = './data/stage1_train'
testpath = './data/stage1_test'
savepath = './data/temp_data'


L_train = LoadImage(trainpath, image_type='train')
    # # preprocessing: -normalization -label contour, inner, outer -overlap mask images for the same target image
image_matrix, mask_matrix = L_train.pile_image_matrix()
    # # save to .npy file
np.save(savepath + '/train_image.npy', image_matrix)
np.save(savepath + '/train_mask.npy', mask_matrix)


L_test = LoadImage(testpath, image_type='test')
image_matrix, mask_matrix = L_test.pile_image_matrix()
np.save(savepath + '/test_image.npy', image_matrix)
