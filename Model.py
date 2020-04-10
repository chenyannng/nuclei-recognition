import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class Model():
    
    def __init__(self):
        pass

    
    def unet(self):
        # build u-net model
        input_shape = (256, 256, 1)
        img_input = tf.keras.Input(shape=input_shape)
        n_kernel = 2** np.array([4, 5, 6, 7, 8])
        # n_kernel = np.array([32,64,128])
        kernel_shape = (3, 3)
        conv = img_input
        list_conv = []
        # encode
        for k in n_kernel:
            conv = Conv2D(k, kernel_shape, activation='relu', padding='same')(conv)
            conv = Dropout(0.2)(conv)
            conv = Conv2D(k, kernel_shape, activation='relu', padding='same')(conv)
            list_conv.append(conv)
            print(conv)
            if k < n_kernel[-1]:
                conv = MaxPooling2D((2, 2))(conv)
            
        # decode
        deconv = conv
        list_conv_rev = list_conv[::-1][1:] # reverse order
        n_kernel_dec = n_kernel[::-1][1:]
        for iq, q in enumerate(n_kernel_dec):
            deconv = Conv2DTranspose(q, (3, 3), activation='relu', padding='same')(deconv)
            deconv = Concatenate(axis=-1)([UpSampling2D((2,2))(deconv), list_conv_rev[iq]])
            deconv = Conv2D(q, (3, 3), activation='relu', padding='same')(deconv)            
            deconv = Dropout(0.2)(deconv)
            deconv = Conv2D(q, (3, 3), activation='relu', padding='same')(deconv)
            deconv = Dropout(0.2)(deconv)

        out = Conv2D(3, (1, 1), padding='same', activation='softmax')(deconv)

        model = tf.keras.Model(img_input, out)

        return model

class Train():

    def __init__(self):
        pass
    

    def train(self, model, x_train, y_train):
        opt = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-5)
        model.compile(optimizer = opt, 
                loss='sparse_categorical_crossentropy', 
                metrics=['sparse_categorical_accuracy'])
        N_EPOCHS = 10
        BATCH_SIZE = 5
        VAL_SPLIT = 0.25
        CALLBACK = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0001, patience=2)
        history = model.fit(x_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs= N_EPOCHS,
                            validation_split=VAL_SPLIT,
                            callbacks=[CALLBACK])
        

        return history