import numpy as np
from config import *
import tensorflow as tf 
from utils import sort_data  
from tensorflow.keras.preprocessing.image import load_img


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self , image_list , mask_list , batch_size , image_size):
        self.batch_size = batch_size
        self.image_list = image_list
        self.mask_list = mask_list
        self.image_size = 160

    def __len__(self):
        return len(self.image_list)//self.batch_size

    def __getitem__(self , index):
        image_list = self.image_list[index*self.batch_size : (index+1)*self.batch_size]
        mask_list = self.mask_list[index*self.batch_size : (index+1)*self.batch_size]

        x = np.zeros((self.batch_size , self.image_size , self.image_size , 3) , dtype = np.float32)
        y = np.zeros((self.batch_size , self.image_size , self.image_size , 1) , dtype = np.float32)

        for file_idx, file in enumerate(image_list):
            img = load_img(file , target_size=(self.image_size , self.image_size))
            x[file_idx] = img

        for file_idx, file in enumerate(mask_list):
            img = load_img(file , target_size=(self.image_size , self.image_size) , color_mode='grayscale')
            y[file_idx] = np.expand_dims(img, 2)
            y[file_idx]-=1

        image=x 
        mask=y

        return image , mask

#test train velidation split


images_dirs,mask_dirs= sort_data(img_dir=img_dirs , mask_dir=mask_dirs)
        
train_img_dirs=images_dirs[:int(len(images_dirs)*0.8)]
train_mask_dirs=mask_dirs[:int(len(mask_dirs)*0.8)]
val_img_dirs=images_dirs[int(len(images_dirs)*0.9):]
val_mask_dirs=mask_dirs[int(len(mask_dirs)*0.9):]
test_img_dirs=images_dirs[int(len(images_dirs)*0.8):int(len(images_dirs)*0.9)]
test_mask_dirs=mask_dirs[int(len(mask_dirs)*0.8):int(len(mask_dirs)*0.9)]

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     dataloader=DataLoader(batch_size=16, image_list=images_dirs, mask_list=mask_dirs,image_size=160)
#     print(len(dataloader))
#     #display the image and mask
#     image ,mask=dataloader[0]
#     #plt.imshow(image[0] / 255)
#     plt.imshow(mask[0] / 255)
#     plt.show()
#     print(image.shape)
#     print(mask.shape)

