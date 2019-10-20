import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import tensorflow.keras.backend as K
import tensorflow as tf
#import segmentation_models as sm

def focal(y_true, y_pred):
    return sm.metrics.f1_score(y_true, y_pred)

def acc(y_true, y_pred):
    gt = K.argmax(y_true, axis=-1)
    pr = K.argmax(y_pred, axis=-1)

    mask = K.greater(gt, 0)
    
    return K.mean(K.equal(tf.boolean_mask(gt,mask), tf.boolean_mask(pr,mask)) )

def acc_full(y_true, y_pred):
    gt = K.argmax(y_true, axis=-1)
    pr = K.argmax(y_pred, axis=-1)

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
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.argmax(y_true, axis=-1), logits=y_pred) #sm.losses.categorical_focal_loss(y_true, K.softmax(y_pred))


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=-1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
