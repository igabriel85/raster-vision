"""
Late Fusion of 2 stream ResNet
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Reshape,
                          Lambda,
                          merge)
import tensorflow as tf

from rastervision.common.models.resnet50 import (ResNet50,
                                                 identity_block,
                                                 conv_block)


DUAL_RESNET = 'dual_resnet'


def make_dual_fcn_resnet(input_shape, dual_active_input_inds,
                         use_pretraining, freeze_base=False,
                         include_top=True, pooling=None,
                         classes=1000, activation='softmax'):
    nb_rows, nb_cols, nb_channels = input_shape
    input_tensor = Input(shape=input_shape)

    # Split input_tensor into two
    def get_input_tensor(it, model_ind):
        # TODO calls to split and concat will need to be updated after
        # upgrading TF
        channel_tensors = tf.split(it, nb_channels, 3)
        input_tensors = []
        for ind in dual_active_input_inds[model_ind]:
            input_tensors.append(channel_tensors[ind])
        input_tensor = tf.concat(input_tensors, 3)
        return input_tensor

    input_tensor1 = Lambda(lambda x: get_input_tensor(x, 0))(input_tensor)
    input_tensor2 = Lambda(lambda x: get_input_tensor(x, 1))(input_tensor)

    # Use weights for either both or none
    weights = 'imagenet' if use_pretraining else None

    base_model1 = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor1)
    for layer in base_model1.layers:
        layer.name += '_1'
    if freeze_base:
        for layer in base_model1.layers:
            layer.trainable = False

    base_model2 = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor2)
    if freeze_base:
        for layer in base_model2.layers:
            layer.trainable = False

    for layer in base_model2.layers:
        layer.name += '_2'

    x_1 = base_model1.get_layer('act4f_1').output

    x_2 = base_model2.get_layer('act4f_2').output

    x = merge([x_1, x_2], mode='concat')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation=activation, name='dense')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)


    model = Model(input_tensor, output=x)

    return model