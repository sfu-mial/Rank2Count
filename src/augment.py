from tensorflow.keras import layers


class RandomImageNoise(layers.Layer):
    """
    RandomImageNoise is a class that applies a random noise to an image.
    Each application applies a shot noise with a random sigma between 0 and sigma.
    This noise is applied to the whole image.
    intensity: the intensity of the image noise
    """
    def __init__(self, intensity, **kwargs):
        super().__init__(**kwargs)
        self.intensity = intensity
    
    def get_config(self):
        config = super().get_config()
        config.update({"intensity": self.intensity})
        return config

    def call(self, images, training=True):
        if training:
            noise = tf.random.uniform(tf.shape(images), minval=-self.intensity, maxval=self.intensity)
            images = tf.clip_by_value(images + noise, 0, 1)
        return images

class AugmentModel(layers.Layer):
    def __init__(self, brightness, contrast, hue, zoomout, rotate, flip, shotnoise, **kwargs):
        super().__init__(**kwargs)
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.zoomout = zoomout
        self.rotate = rotate
        self.flip = flip
        self.shotnoise = shotnoise

        self.brightness_layer = layers.RandomBrightness(**self.brightness)
        self.contrast_layer = layers.RandomContrast(**self.contrast)
        self.hue_layer = layers.RandomHue(**self.hue) # TODO
        self.zoomout_layer = layers.RandomZoom(**self.zoomout)
        self.rotate_layer = layers.RandomRotation(**self.rotate)
        self.flip_layer = layers.RandomFlip(**self.flip)
        self.shotnoise_layer = RandomImageNoise(**self.shotnoise)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'brightness': self.brightness,
            'contrast': self.contrast,
            'hue': self.hue,
            'zoomout': self.zoomout,
            'rotate': self.rotate,
            'flip': self.flip,
            'shotnoise': self.shotnoise,
        })
        return config
    
    def call(self, images, training=True):
        if training:
            images = self.brightness_layer(images)
            images = self.contrast_layer(images)
            images = self.hue_layer(images)
            images = self.zoomout_layer(images)
            images = self.rotate_layer(images)
            images = self.flip_layer(images)
            images = self.shotnoise_layer(images)
        return images