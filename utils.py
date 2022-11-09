import os 
from config import * 
import matplotlib.pyplot as plt
import numpy as np


def sort_data(img_dir , mask_dir):
    image_list = []
    for file in os.listdir(img_dir):
        if file.endswith(".jpg"):
            path='C:/Users/Amzad/Desktop/segmentation/data/images/'+file
            image_list.append(path)

    mask_list = []
    for file in os.listdir(mask_dir):
        if file.endswith(".png") and not file.startswith('.'):
            path='C:/Users/Amzad/Desktop/segmentation/data/annotations/trimaps/'+file
            mask_list.append(path)

    return sorted(image_list) ,sorted(mask_list)


x,y = sort_data(img_dir=img_dirs , mask_dir=mask_dirs) 


def plot_img_mask_prediction(data=val_ds):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            image, mask = data[i * 3 + j]
            pred_mask = model.predict(image[tf.newaxis, ...])
            ax[i, j].imshow(image)
            ax[i, j].imshow(mask[:, :, 0], alpha=0.5, cmap='gray')
            ax[i, j].imshow(np.argmax(pred_mask[0], axis=-1), alpha=0.5, cmap='gray')
            ax[i, j].axis('off')
    plt.show()

