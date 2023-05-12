from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, ReLU, LeakyReLU, Dense, GlobalAveragePooling2D, add, subtract, Flatten, Layer, Sequential
import tensorflow as tf

def downsample(filters, size, apply_batchnorm=True, padding='same'):
    """
    Downsampling Block for GAN, that takes in a tensor of shape (H, W, C) and produced a new tensor of shape (0.5H, 0.5W, C)
    borrowed from https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding=padding,
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    """
    Upsampling block for GANs, that takes in a tensor of shape (H, W, C) and produced a new tensor of shape (2H, 2W, C)
    borrowed from https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.2))
    result.add(ReLU())

    return result

def base_discriminator(bnorm=False):
    return Sequential([Input((56,56,1)), tf.keras.layers.experimental.preprocessing.RandomFlip(),
                                 downsample(32, 3, padding='valid'), downsample(128, 3, bnorm, padding='valid'),
                                 downsample(256, 3, bnorm, padding='valid'), downsample(256, 3, bnorm, padding='valid'),
                                 GlobalAveragePooling2D(), Dropout(0.2),
                                 Dense(256, activation="relu"), Dense(128, activation="relu"),
                                 Dense(1, activation=None)], name="discriminator")

def base_feat_extract(weights):
    outputs = ["conv2_block3_1_relu", "conv3_block4_1_relu", "conv4_block6_1_relu", "post_relu"]
    resnet = ResNet50V2(include_top=False, pooling="avg", weights=weights, input_shape=(224,224,3))
    return Model(resnet.inputs, [resnet.get_layer(o).output for o in outputs], name="resnet_features")

def density_map_generator():
    """
    Builds the density map generator that takes in a tensor of shape (224, 224, 3) and produces a density map of shape (56, 56, 1)
    """
    feat56, feat28, feat14, feat7 = Input([56,56,64]), Input([28,28,128]), Input([14,14,256]), Input([7,7,2048])
    x = Conv2D(256, 1, padding="SAME", activation="relu")(feat7)
    for f, feat in zip([256, 128, 64], [feat14, feat28, feat56]):
        x = upsample(f, 4, True)(x)
        feat = Conv2D(f, 4, padding="SAME", activation="relu")(feat)
        x = add([feat, x])
    x = Conv2D(32, 4, padding="SAME", activation="relu")(x)
    dmap_pred = Conv2D(1, 4, padding="SAME", activation="sigmoid")(x)
    return Model([feat56, feat28, feat14, feat7], dmap_pred, name="generator")

class SumLayer(Layer):
    """
    Custom layer that sums over a given axis
    """
    def __init__(self, axis, keepdims=False):
        super(SumLayer, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.keras.backend.sum(inputs, axis=self.axis, keepdims=self.keepdims)
    
def build_rank_head():
    """
    Builds the rank head that takes in two density maps, sums over them, and outputs a scalar value
    representing the difference between the two density maps
    """
    dmap_i = Input([56, 56, 1])
    dmap_j = Input([56, 56, 1])
    c_i = SumLayer(1, True)(Flatten()(dmap_i))
    c_j = SumLayer(1, True)(Flatten()(dmap_j))
    
    diff_ij = subtract([c_i, c_j])*0.3
    return Model([dmap_i, dmap_j], diff_ij, name="rank_head")