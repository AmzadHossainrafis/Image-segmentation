from keras.models import load_model
from config import *
from matrices import *
from dataloader import Dataloader,test_img_dirs,test_mask_dirs
#from matrices import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import prediction
from tensorflow.keras.preprocessing.image import load_img


metrics = ['acc','mse','mae','mape',cat_acc,iou_coef]

model=load_model(load_model_dir, custom_objects={'cat_acc':cat_acc,'iou_coef':iou_coef})
model.compile(optimizer=keras.optimizers.Adam(learning_rate),loss=loss_name,metrics=metrics)


train_ds=Dataloader()

history1=model.evaluate(train_ds,)