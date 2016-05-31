import numpy as np
import tensorflow as tf

def fill_feed_dict(features, weights, feature_values, weight_values):
  rows = len(feature_values)
  feed_dict= {
      features: np.reshape(feature_values, (rows, 1)),
      weights: np.reshape(weight_values, (1, 1))
  }
  return feed_dict

def predict_op(feature_values, weight_values):
  features = tf.placeholder(tf.float32, shape=[None,1])
  weights = tf.placeholder(tf.float32, shape=[1, None])
  return tf.matmul(features, weights)

def predict(feature_values, weight_values):
  feed_dict= fill_feed_dict(features, weights, feature_values, weight_values)
  sess = tf.Session()
  return sess.run(predict_op, feed_dict = feed_dict)

def error(feature_values, weight_values, actual_values):
  expected = tf.placeholder(tf.float32, shape=[None,1])  
  difference = expected - predict_op(feature_values, weight_values)
  return tf.reduce_sum(tf.square(difference))
  
  
