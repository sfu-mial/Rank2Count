import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from skimage.filters import gaussian
import numpy as np


@tf.function
def parse_images(image_path, size):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[size[0], size[1]])
    return image

def get_count_image_ds(images, labels, size, shuffle=False):
    """
    get_count_image_ds is a function that takes a list of image paths and 
    returns a dataset of images and their count labels.
    """
    image_ds = Dataset.from_tensor_slices(images)
    label_ds = Dataset.from_tensor_slices(labels)
    count_ds = Dataset.zip((image_ds, label_ds))
    if shuffle:
        count_ds = count_ds.shuffle(6000)
    count_ds = count_ds.map(lambda x, y: (parse_images(x, size), y), num_parallel_calls=AUTOTUNE)
    return count_ds

def get_paired_image_ds(im_src, im_tar, labels, size, shuffle=False):
    """
    get_paired_image_ds is a function that takes two lists of image paths and 
    returns a dataset of paired images and their ordering labels.
    """
    src_ds = Dataset.from_tensor_slices(im_src)
    tar_ds = Dataset.from_tensor_slices(im_tar)
    label_ds = Dataset.from_tensor_slices(labels)
    rank_ds = Dataset.zip((src_ds, tar_ds))
    rank_ds = Dataset.zip((rank_ds, label_ds))
    if shuffle:
        rank_ds = rank_ds.shuffle(6000)
    # map the rank_ds to parse_images, and returns a dataset (tar, src, label)
    rank_ds = rank_ds.map(lambda x, y: (parse_images(x[0], size), parse_images(x[1], size), y),
                          num_parallel_calls=AUTOTUNE)
    return rank_ds

def sample_gaussian_density_map(index):
    p = np.random.choice(range(5,80), size=1)[0]/(56*56)
    dmap1 = np.random.choice([0,1], p=[1-p,p], size=(56,56))
    densitymap1 = gaussian(dmap1.astype(np.float64), sigma=(1.5,1.5), mode="reflect")
    densitymap1 = np.expand_dims(densitymap1, axis=-1)
    return densitymap1

def tf_density_map(index):
    im_shape = (56,56,1)
    [dmap1,]= tf.py_function(sample_gaussian_density_map, [index], [tf.float32])
    dmap1.set_shape(im_shape)
    return dmap1

def get_density_ds(size):
    dmap_ds = Dataset.from_tensor_slices(np.arange(0,size))
    dmap_ds = dmap_ds.map(tf_density_map, num_parallel_calls=AUTOTUNE)
    return dmap_ds