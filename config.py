img_dirs =r'C:/Users/Amzad/Desktop/segmentation/data/images/'
mask_dirs =r'C:\Users/Amzad/Desktop/segmentation/data/annotations/trimaps/'
weights_dir=r'C:\Users/Amzad\Desktop/segmentation/project/logs/weights/'

img_size = (160, 160)
model_nam='unet'
batch_size=16
epochs=10
height = 160
width = 160
num_classes = 3
channels=3
early_stop="True"
list_matrices=[]
loss_name="categorical_crossentropy"
learning_rate=1e-3
mix_precision="False"