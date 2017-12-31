from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
MUSHROOM_DATA = "mushrooms.csv"

feature_descriptor_dict = {}
feature_descriptor_dict["cap-shape"] = {
    "b": "bell",
    "c": "conical",
    "x": "convex",
    "f": "flat",
    "k": "knobbed",
    "s": "sunken"
}

feature_descriptor_dict["cap-surface"] = {
    "f": "fibrous",
    "g": "grooves",
    "y": "scaly",
    "s": "smooth"
}

feature_descriptor_dict["cap-color"] = {
    "n": "brown",
    "b": "buff",
    "c": "cinnamon",
    "g": "gray",
    "r": "green",
    "p": "pink",
    "u": "purple",
    "e": "red",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["bruises"] = {"t": "bruises", "f": "no"}

feature_descriptor_dict["odor"] = {
    "a": "almond",
    "l": "anise",
    "c": "creosote",
    "y": "fishy",
    "f": "foul",
    "m": "musty",
    "n": "none",
    "p": "pungent",
    "s": "spicy"
}

feature_descriptor_dict["gill-attachment"] = {
    "a": "attached",
    "d": "descending",
    "f": "free",
    "n": "notched"
}

feature_descriptor_dict["gill-spacing"] = {"close": "c", "crowded": "w", "distant": "d"}

feature_descriptor_dict["gill-size"] = {"broad": "b", "narrow": "n"}

feature_descriptor_dict["gill-color"] = {
    "k": "black",
    "n": "brown",
    "b": "buff",
    "h": "chocolate",
    "g": "gray",
    "r": "green",
    "o": "orange",
    "p": "pink",
    "u": "purple",
    "e": "red",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["stalk-shape"] = {"enlarging": "e", "tapering": "t"}

feature_descriptor_dict["stalk-root"] = {
    "b": "bulbous",
    "c": "club",
    "u": "cup",
    "e": "equal",
    "z": "rhizomorphs",
    "r": "rooted",
    "?": "missing"
}

feature_descriptor_dict["stalk-surface-above-ring"] = {
    "f": "fibrous",
    "y": "scaly",
    "k": "silky",
    "s": "smooth"
}

feature_descriptor_dict["stalk-surface-below-ring"] = {
    "f": "fibrous",
    "y": "scaly",
    "k": "silky",
    "s": "smooth"
}

feature_descriptor_dict["stalk-color-above-ring"] = {
    "n": "brown",
    "b": "buff",
    "c": "cinnamon",
    "g": "gray",
    "o": "orange",
    "p": "pink",
    "e": "red",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["stalk-color-below-ring"] = {
    "n": "brown",
    "b": "buff",
    "c": "cinnamon",
    "g": "gray",
    "o": "orange",
    "p": "pink",
    "e": "red",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["veil-type"] = {"partial": "p", "universal": "u"}

feature_descriptor_dict["veil-color"] = {
    "n": "brown",
    "o": "orange",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["ring-number"] = {"none": "n", "one": "o", "two": "t"}

feature_descriptor_dict["ring-type"] = {
    "c": "cobwebby",
    "e": "evanescent",
    "f": "flaring",
    "l": "large",
    "n": "none",
    "p": "pendant",
    "s": "sheathing",
    "z": "zone"
}

feature_descriptor_dict["spore-print-color"] = {
    "k": "black",
    "n": "brown",
    "b": "buff",
    "h": "chocolate",
    "r": "green",
    "o": "orange",
    "u": "purple",
    "w": "white",
    "y": "yellow"
}

feature_descriptor_dict["population"] = {
    "a": "abundant",
    "c": "clustered",
    "n": "numerous",
    "s": "scattered",
    "v": "several",
    "y": "solitary"
}

feature_descriptor_dict["habitat"] = {
    "g": "grasses",
    "l": "leaves",
    "m": "meadows",
    "p": "paths",
    "u": "urban",
    "w": "waste",
    "d": "woods"
}

TARGET_LABEL = "class"
target_descriptor_dict = {"e": "edible", "p": "poisonous"}
target_index_dict = {"e": 0, "p": 1}

TRAINING_TEST_DATA_SPLIT = 0.8

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({k: data_set[k].values for k in feature_descriptor_dict.keys()}),
    y=pd.Series(data_set[TARGET_LABEL].values).map(target_index_dict),
    num_epochs=num_epochs,
    shuffle=shuffle)

def main(_):
  # Load datasets.
  mushroom_data = pd.read_csv(MUSHROOM_DATA, skipinitialspace=True)

  # Randomly select which entries of data should be used for training and testing.
  random_selection_mask = np.random.rand(len(mushroom_data)) < TRAINING_TEST_DATA_SPLIT
  training_set = mushroom_data[random_selection_mask]
  test_set = mushroom_data[~random_selection_mask]

  # Generate TensorFlow feature cols
  feature_column_dict = {}
  for feature, value_dict in feature_descriptor_dict.iteritems():
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        feature, value_dict.keys())
    feature_column_dict[feature] = tf.feature_column.indicator_column(categorical_column) 

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(
      feature_columns=feature_column_dict.values(),
      hidden_units=[10, 20, 10],
      model_dir="/tmp/mishroom_model")
  
  # Train.
  regressor.train(get_input_fn(training_set, num_epochs=1, shuffle=False), steps=2000)
  
  # Evaluate loss and accuracy over one epoch of test_set.
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

if __name__ == "__main__":
  tf.app.run()
