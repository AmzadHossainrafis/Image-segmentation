
from config import *
import segmentation_models as sm
import keras_unet_collection.models as kuc












def model_chg(model_name=model_nam,height=height,width=width,channels=channels,classes=num_classes):
    if model_name=='attention_unet':
        model = kuc.att_unet_2d((height, width, channels), filter_num=[64, 128, 256, 512, 1024], n_labels=classes)
        # uncomment if you want to use mix precision
        # x = model.layers[-2].output # fetch the last layer previous layer output
        # output = Conv2D(num_classes, kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        # model = Model(inputs = model.input, outputs=output)
        return model

    elif model_name=='unet':
        model = sm.Unet(backbone_name='efficientnetb0', input_shape=(height, width, channels),
                            classes = num_classes, activation='softmax',
                            encoder_weights=None, weights=None)
        # uncomment if you want to use mix precision
        # x = model.layers[-2].output # fetch the last layer previous layer output
        # output = Conv2D(num_classes, kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        # model = Model(inputs = model.input, outputs=output)
        return model

    elif model_name=='vnet':
            model = kuc.vnet_2d((height,width,channels), filter_num=[16, 32, 64, 128, 256], 
                        n_labels=num_classes ,res_num_ini=1, res_num_max=3, 
                        activation='PReLU', output_activation='Softmax', 
                        batch_norm=True, pool=False, unpool=False, name='vnet')
    # uncomment if you want to use mix precision
    # x = model.layers[-2].output # fetch the last layer previous layer output
    # output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    # model = Model(inputs = model.input, outputs=output)
            return model

    elif model_name=="link_net":
         model = sm.Linknet(backbone_name='efficientnetb0', input_shape=(height, width, channels),
                    classes = num_classes, activation='softmax',
                    encoder_weights=None, weights=None)

         # x = model.layers[-2].output # fetch the last layer previous layer output
         # output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
         # model = Model(inputs = model.input, outputs=output)
         return model     
    
    elif model_name=="pspnet":
        model = sm.PSPNet(backbone_name='efficientnetb0', input_shape=(height, width, channels),
                    classes =num_classes, activation='softmax',
                    encoder_weights=None, weights=None)
        # x = model.layers[-2].output # fetch the last layer previous layer output

        # output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        # model = Model(inputs = model.input, outputs=output)
        return model