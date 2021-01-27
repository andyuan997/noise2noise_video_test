import argparse
import numpy as np
from pathlib import Path
import cv2

from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

import time


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    image_dir = "./inputdir"
    weight_file = "./weights.hdf5"

    vedio_dir = "test_video.DAT"
    
    model = get_srresnet_model()
    model.load_weights(weight_file)

    image_paths = list(Path(image_dir).glob("*.*"))

    
    
    #影片
    cap = cv2.VideoCapture(vedio_dir)
    cap2 = cv2.VideoCapture(vedio_dir)
    
    while (cap.isOpened()):

        t0 = time.time()
        ret, frame = cap.read()
        ret, frame2 = cap2.read()

        #改變大小
        #frame = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_CUBIC)
        #frame2 = cv2.resize(frame2, (600, 400), interpolation=cv2.INTER_CUBIC)

        
        cv2.imshow('input', frame)

        pred = model.predict(np.expand_dims(frame2, 0))
        denoised = get_image(pred[0])
        cv2.imshow('output', denoised)
        
        t1 = time.time()
        print('time(s):',t1-t0)
        print('FPS:',1/(t1-t0))
        
        #time.sleep(0.03)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    

    '''
    #圖片    

    for image_path in image_paths:
        t0 = time.time()
        image = cv2.imread(str(image_path))

        #改變大小
        #image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_CUBIC)
        
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        h, w, _ = image.shape
        
        out_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
        #out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        noise_image = image
        #print(type(noise_image))
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])
        out_image[:, :w] = image
        
        #out_image[:, w:w * 2] = noise_image
        #out_image[:, w * 2:] = denoised_image
        
        out_image[:, w:w * 2] = denoised_image

        cv2.imshow("result", out_image)
        t1 = time.time()

        print(t1-t0)
        print(1/(t1-t0))
        
        key = cv2.waitKey(-1)
        # "q": quit
        if key == 113:
            return 0
        '''
       

def get_srresnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


if __name__ == '__main__':
    main()
