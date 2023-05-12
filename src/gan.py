import tensorflow as tf
from src.model import base_feat_extract, density_map_generator, base_discriminator, build_rank_head, SumLayer
from tensorflow.keras.layers import *
from src.losses import disc_advr_loss, gen_advr_loss
from src.augment import AugmentModel

class DMapGAN(tf.keras.Model):
    def __init__(self, weights="imagenet"):
        super(DMapGAN, self).__init__()
        self.GetFeat = base_feat_extract(self.weights)
        self.Genr = density_map_generator()
        self.Disc = base_discriminator()
        self.Rank = build_rank_head()
        self.sum = SumLayer(1, True)
        self.flat = Flatten()
        self.Augm = AugmentModel(brightness = 0.10, contrast=0.3, hue=0.5, 
                                 zoomout=((0, 0.40), None, "constant"), 
                                 rotate=(0.02, "constant"), flip="horizontal", shotnoise=0.07)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, r_loss_fn, g_loss_metric,
                d_loss_w=1, g_loss_w=1, r_loss_w=1):
        super(DMapGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.rank_loss_fn = r_loss_fn

        self.d_loss_w = d_loss_w
        self.g_loss_w = g_loss_w
        self.r_loss_w = r_loss_w

        self.g_loss_metric = g_loss_metric

    @tf.function
    def basic_step(self, img_i, img_j, dmap_real, training):

        img_i_aug = self.Augm(img_i, training=training)
        img_j_aug = self.Augm(img_j, training=training)

        feats_i = self.GetFeat(img_i_aug, training=training)
        feats_j = self.GetFeat(img_j_aug, training=training)

        dmap_fake_i = self.Genr(feats_i, training=training)
        dmap_fake_j = self.Genr(feats_j, training=training)

        rank_ij = self.Rank([dmap_fake_i, dmap_fake_j], training=training)

        adv_D_real = self.Disc(dmap_real)
        adv_D_fake = concatenate([self.Disc(dmap_fake_i, training=training), self.Disc(dmap_fake_j, training=training)], axis=0)
        return adv_D_real, adv_D_fake, rank_ij

    def train_step(self, data):
        img_i, img_j, rank_ij_gt, dmap_real = data
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            adv_D_real, adv_D_fake, rank_ij = self.basic_step(img_i, img_j, dmap_real, training=True)
            g_loss = self.g_loss_w*gen_advr_loss(self.g_loss_fn, adv_D_fake)
            d_tot_loss = self.d_loss_w*disc_advr_loss(self.d_loss_fn, adv_D_real, adv_D_fake)
            r_loss = self.r_loss_w*self.rank_loss_fn(rank_ij_gt, rank_ij)
            g_tot_loss = g_loss + r_loss
        
        disc_weights = self.Disc.trainable_weights
        genr_weights = self.Genr.trainable_weights + self.GetFeat.trainable_weights + self.Rank.trainable_weights
        
        d_grads = D_tape.gradient(d_tot_loss, disc_weights)
        g_grads = G_tape.gradient(g_tot_loss, genr_weights)

        self.d_optimizer.apply_gradients(zip(d_grads, disc_weights))
        self.g_optimizer.apply_gradients(zip(g_grads, genr_weights))

        return {"d_loss": d_tot_loss, "g_loss": g_tot_loss, "r_loss": r_loss}
    
    def test_step(self, data):
        # Provides a custom testing step, but only for the generator
        img, cnt_gt = data
        feats = self.GetFeat(img, training=False)
        dmap = self.Genr(feats, training=False)
        # sum over the dmap
        cnt_pred = self.sum(self.flat(dmap))
        g_cnt_loss = self.g_loss_metric(cnt_gt, cnt_pred)
        return {"g_cnt_loss": g_cnt_loss}

    def call(self, inputs):
        return self.Genr(self.GetFeat(inputs, training=False), training=False)