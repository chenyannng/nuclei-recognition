from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from Model import Model, Train
from DataAugmentation import DataAugmentation
from Plots import Plots


if __name__ == "__main__":

    # prepare training data
    savepath = './temp_data'
    image_matrix = np.load(savepath + '/train_image.npy')
    mask_matrix = np.load(savepath + '/train_mask.npy')
    image_matrix = np.reshape(image_matrix, image_matrix.shape + (1,)) # add one dimension at tuple end, for model input purpose
    mask_matrix = np.reshape(mask_matrix, mask_matrix.shape + (1,))

    
    # augmentation
    print('Doing augmentation:...')
    A_img = DataAugmentation(image_matrix)
    aug_image_matrix = A_img.aug_array
    A_msk = DataAugmentation(mask_matrix)
    aug_mask_matrix = A_msk.aug_array
    print(aug_image_matrix.shape)
    
    test_img = np.load(savepath + '/test_image.npy')
    test_img = np.reshape(test_img, test_img.shape + (1,))
    np.save(savepath + '/test_image00.npy', test_img)


    model = Model().unet()
  
    print('Model structure saved as png.')
    tf.keras.utils.plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    train_history = Train().train(model, aug_image_matrix, aug_mask_matrix)   
    print('\nhistory dict:', train_history.history)

    # plot train history 

    plot = Plots(train_history)
    plot.plot_metrics()
    
    print('\n# Generate predictions for samples')
    predictions = model.predict(test_img)
    print('Saved predictions data:', predictions.shape)
    # write predictions to file
    savepath = 'D:/Code/convnet/temp_data'
    np.save(savepath + '/test_pred.npy', predictions)

    
