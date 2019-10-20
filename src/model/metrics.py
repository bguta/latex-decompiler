import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import tensorflow.keras.backend as K
import tensorflow as tf
#import segmentation_models as sm

def focal(y_true, y_pred):
    return sm.metrics.f1_score(y_true, y_pred)

def acc(y_true, y_pred):
    pr = K.argmax(K.softmax(y_pred), axis=-1)
    gt = K.cast(y_true, 'int64')
    return K.mean(K.equal(gt, pr))

def acc_full(y_true, y_pred):
    gt = K.argmax(y_true, axis=-1)
    pr = K.argmax(K.softmax(y_pred), axis=-1)

    mask = K.greater(gt, 0)
    
    return K.mean(K.equal(gt, pr))

# def acc(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     return tf.keras.metrics.sparse_categorical_accuracy(K.argmax(y_true, axis=-1), K.softmax(y_pred))#sm.metrics.recall(y_true, K.softmax(y_pred))

def ce(y_true, y_pred):
    gt = K.cast(y_true, 'int32')
    return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.squeeze(gt,axis=-1), logits=y_pred)) #sm.losses.categorical_focal_loss(y_true, K.softmax(y_pred))

