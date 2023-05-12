import tensorflow as tf

def disc_advr_loss(loss_obj, adv_D_pred_real, adv_D_pred_fake):
    adv_L_real =  loss_obj(tf.ones_like(adv_D_pred_real), adv_D_pred_real)
    adv_L_fake =  loss_obj(tf.zeros_like(adv_D_pred_fake), adv_D_pred_fake)
    adv_L = adv_L_real + adv_L_fake
    return adv_L

def gen_advr_loss(loss_obj, adv_D_pred_fake):
    adv_L = loss_obj(tf.ones_like(adv_D_pred_fake), adv_D_pred_fake)
    return adv_L