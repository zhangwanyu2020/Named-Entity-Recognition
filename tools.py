def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return micro_f1, macro_f1


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


def get_weight(weight_1,weight_0):
    def mycrossentropy(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
        loss = weight_1*K.binary_crossentropy(y_true, y_pred)*pt_1+weight_0*K.binary_crossentropy(y_true, y_pred)*pt_0
        return loss
    return mycrossentropy
