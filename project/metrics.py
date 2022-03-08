from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# Keras MeanIoU
# ----------------------------------------------------------------------------------------------

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)

def cat_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true,y_pred)


'''
meanIOU
class acc separate
dice cof
f1 accuracy
auc
'''
# Matrics
# ----------------------------------------------------------------------------------------------

def get_metrics(config):
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    
    m = MyMeanIOU(config['num_classes'])
    return {
            'cat_acc':cat_acc,
            'MyMeanIOU': m
          }
#metrics = ['acc']