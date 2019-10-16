import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import tensorflow.keras.backend as K
import tensorflow as tf
import segmentation_models as sm

def focal(y_true, y_pred):
    return sm.metrics.f1_score(y_true, y_pred)

def acc(y_true, y_pred):
    
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

# def acc(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     return tf.keras.metrics.sparse_categorical_accuracy(K.argmax(y_true, axis=-1), K.softmax(y_pred))#sm.metrics.recall(y_true, K.softmax(y_pred))

def ce(y_true, y_pred):
    return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.argmax(y_true, axis=-1), logits=y_pred))

