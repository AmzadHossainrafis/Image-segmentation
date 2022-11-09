from model import model_chg 
from config import *

import tensorflow as tf
import keras
from dataloader import DataLoader,train_img_dirs,train_mask_dirs,val_img_dirs,val_mask_dirs
#from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from keras.callbacks import ModelCheckpoint

train_ds=DataLoader(batch_size=32,image_list=train_img_dirs,mask_list=train_mask_dirs,image_size=160)
val_ds=DataLoader(batch_size=32,image_list=val_img_dirs,mask_list=val_mask_dirs,image_size=160)



model=model_chg(model_name=model_nam,height=height,width=width,channels=channels,classes=num_classes)
chk_point=ModelCheckpoint(weights_dir+"/{}_date.hdf5".format(model_nam),save_best_only=False)


callbacks = [
   chk_point
]
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# Train the model, doing validation at the end of each epoch.
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

# uncomment this line to display prediction
# fig, ax = plt.subplots(3, 3, figsize=(10, 10))
# image, mask = val_ds[0]
# for i in range(3):

#     pre=image[i+1]
#     exp=np.expand_dims(pre,axis=0)   
#     pred_mask = model.predict(exp)
#     #title image
#     ax[i, 0].set_title('Image')
#     ax[i, 0].imshow(image[i+1]/255)
#     ax[i,0].axis('off')
#     #title mask
#     ax[i, 1].set_title('Ground truth')
#     ax[i, 1].imshow(mask[i+1])
#     ax[i,1].axis('off')
#     #title prediction
#     ax[i, 2].set_title('Prediction')
#     ax[i, 2].imshow(np.argmax(pred_mask[0], axis=-1) , cmap='gray')
#     ax[i,2].axis('off')
#     #save the image
#     plt.savefig('C:/Users/Amzad/Desktop/segmentation/project/logs/weights/{}.png'.format(model_nam))