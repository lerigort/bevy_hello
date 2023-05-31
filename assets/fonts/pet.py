# -*- coding: utf-8 -*-
"""Pet.ipynb


!!!I HIGHLY RECOMMEND TO LOOK AT THE ORIGINAL FILE, WITH PREPARED RESULTS!!!

Original file is located at
    https://colab.research.google.com/drive/1NGpeYLAeO7avFIAlPwziD57LPcO3-YgG
    

In this demo we going to predict quality of wine by 6-8 features. 
It's going to be different data type inputs, such as:

Description - natural language data;

Country - categorical;

Price - numerical.

For every input we should prepare and adapt individual encoder (which is the hardest part of project)

**Import and read**
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers
import seaborn as sn

# Download and read the data.

url = "https://drive.google.com/u/0/uc?id=1fakbbdmqXLY8cyQdbd2kbHDQaN_XzCQb&export=download"

df = pd.read_csv(url)

df.tail()

"""**Treat NaN instances.**"""

# Drop definetly unimportant labels.
df = df.drop(['Unnamed: 0', 'region_1', 'region_2'], axis=1)
df.tail()

# There is a lot unfilled values - count NaN nodes in every column.
def count_nan(df):
  for column in df.columns:
    all_nan = df[column].isna().sum()
    print(column, ' = ', all_nan)

count_nan(df)

# Usefull feature to look at rows with NaN instances.
df[df['country'].isna()]

# Drop NaN rows from 'country', 'variety' and 'province' labels. 
# It's about 130 instances - relatively small amount of data.

df = df.dropna(subset=['country', 'variety', 'province'])
df.tail()

# Last index is stil 129970, which is no good.
# But, as we can see, NaN values were sucsesfully dropped:
count_nan(df)

# We need to reset indexes.
df = df.reset_index(drop=True)
df.tail()

# There is still alot of instances with NaNs - at least 37453.
# We can count the precise cross amount, and implement some workaround.
# But I'd rather drop the labels - model already unnecessary complex.
# Even without additional labels tuning it will be a difficult task.

df.drop(['designation', 'taster_name', 'taster_name', 'taster_twitter_handle'], axis=1, inplace=True)
count_nan(df)

# We going to predict 'Price' - have to clear it too form NaNs.

df.dropna(subset='price', inplace=True)
df = df.reset_index(drop=True) # Do not forget reset index.

df.tail()

# Data prepared for analysis
count_nan(df)

"""**Analysis of features - how to treat them (as a categorical data or as a text)**"""

# Now its time to decide how to treat these features: 
# 'Province',	'title', 'variety', 'winery'.

# If diversity isn't too high, we can treat them as categorical data.
# Count unique features.

labels = ['province',	'title',	'variety',	'winery']

for label in labels:
  unique = df[label].unique()
  print(f'Unique features in {label} = {len(unique)}')

# Even lowest 422 unique features seems too much.
# But, maybe there is 10 or 50 main 'provinces' or 'varieties' -
# - which contains like 80% of data.

# Let's count:

# You can just skip this part of code - lets focus on the analysis.
def count_top_unique(df, label, top_count=30):
  all_unique_amounts={}

  for unique in df[label].unique():
    feature_amount = len(df[df[label] == unique])
    all_unique_amounts[unique] = feature_amount

  all_unique_amounts = dict(sorted(all_unique_amounts.items(), key=lambda item: item[1], reverse=True)) # descending sort by values
  top_unique = list(all_unique_amounts.values())[:top_count] # slice first 30 unique features

  # and lets show the data
  print(f'There is {sum(top_unique)} of {len(df)} instances in top {top_count} unique features')
  print(f'Or, {int(sum(top_unique)/len(df)*100)}% of instances in top {top_count} features')

count_top_unique(df, 'province')

count_top_unique(df, 'variety')

# I'd like to have plot, so we should modify this function a bit.

# We're going to return a list of values:
# [amount of data in top 10 features, top 20, top 30, ...] etc.
# NOT top 10% - it's too much.
def top_unique_for_plot(df, label) -> list[int]:
  top_count = range(10, 100, 10)
  all_unique_amounts=[]
  
  for unique in df[label].unique():
    feature_amount = len(df[df[label] == unique])
    all_unique_amounts.append(feature_amount)

  all_unique_amounts = sorted(all_unique_amounts, reverse=True)

  top_unique = []
  for count in top_count:
    sum_of_top = sum(all_unique_amounts[:count])
    # we want percents of all data, it's more handy for plot
    top_unique.append(sum_of_top / len(df))
  
  return top_unique

def plot_top_data(df, label):
  top_unique = top_unique_for_plot(df, label)
  
  plt.plot(top_unique, range(10, 100, 10), color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
  plt.ylabel('top X unique features')
  plt.xlabel('% of data in top features')
  plt.title(f'Percent of data in top X features in {label} label') 
  plt.show()

# In top [10, 20, 30 ...] features collected [55%, 73%, 87%...] of data.
plot_top_data(df, 'variety')

plot_top_data(df, 'province')

# As we can see, 80%+ of data stored under top 30-40 unique features.
# I'll pick up top 50 features - just to be sure.

# So we actually can make it categorical.
# Just have to treat other 400 unique features as 'OutOfLibrary' data.

# 'Title' and	'winery' features, where 15k and 100k+ unique values, I'll treat as a text data.

# Now we need to prepare our data before feeding the model.

"""!!!FROM THIS PLACE YOU CAN SKIP RIGHT TO THE "MODEL" SECTION!!!

SPLIT THE DATA \ CONVERT DF -> DS
"""

# For training we take 80% of DF, 10% for test, 10% for validation. 
# There 100k+ instances, so 10% for each is enought for sure.

train, test, val = np.split(df.sample(frac=1), [int(len(df)*0.8), int(len(df)*0.9)])

# We should transform our dataframes to `tf.data.Dataset` before encoding
# Otherwise we'll catch nasty dimensional errors.

def df_to_dataset(dataframe, label, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop(label)

  # We got ds/dict kind of dataset - acsess to features throught the keys.
  df = {key: tf.expand_dims(value, axis=-1) for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

# All DF lables would be in dict format as feature in DS
# And 'points' would be label in DS

train_ds = df_to_dataset(df,  label='points')
test_ds = df_to_dataset(df,  label='points')
val_ds = df_to_dataset(df, label='points')

# Let's look inside of batch of dataset.
[(train_sample_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_sample_features.keys()))
print('A batch of feature:', train_sample_features['winery'])
print('A batch of targets:', label_batch)

"""**PREPROCESSING \ ENCODING**

NORMALIZATION
"""

# We need to encode different sorts of data before feeding the model
# Now we going to prepare encoding layers, and then - apply them to model as the input layers

# We need normalization for numerical data - as prices.

def get_normalization_layer(name, dataset):
  
  # Create a Normalization layer for the feature.
  normalizer = tf.keras.layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

price_normalization_layer = get_normalization_layer('price', train_ds)

# Check, how our encoding layer works.
[(train_sample_features, label_batch)] = train_ds.take(1)
price_normalization_layer(train_sample_features['price'])

"""CATEGORIZATION"""

# Keras only supports one-hot-encoding for data that has already been integer-encoded. 
# We need pre integer-encode our strings:

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

# Let's check encoding layer:

country_categorization_layer = get_category_encoding_layer(name='country',
                                              dataset=train_ds,
                                              dtype='string')

country_categorization_layer(train_sample_features['country'])

province_categorization_layer = get_category_encoding_layer(name='province',
                                              dataset=train_ds,
                                              dtype='string',
                                              max_tokens=50)

province_categorization_layer(train_sample_features['province'])

variety_categorization_layer = get_category_encoding_layer(name='variety',
                                              dataset=train_ds,
                                              dtype='string',
                                              max_tokens=50)

variety_categorization_layer(train_sample_features['variety'])

"""TEXT VECTORIZATION"""

# We need to prepare our natural language data.
# For that perfectly fits TensorFlow Hub_layer
# This is already trained model for vectorization of natural language.

embedding = "https://tfhub.dev/google/nnlm-en-dim128/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

# It won't work with higher dimensions, so we need to squeeze DS.
hub_layer(tf.squeeze(train_sample_features['description']))

# Specific names of wineries and titles of wines doesn't look as natural language data
# But actual hub-encoding looks pretty nice (and basical string encoding looks like a mess)

hub_layer(tf.squeeze(train_sample_features['winery']))

embedding_64 = "https://tfhub.dev/google/nnlm-en-dim64"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

# Prepare function
text_encoding = lambda text: hub_layer(tf.squeeze(text))

"""STRING ENCODING"""

# We not going to use string encoding, 
# But I'll leave this part of code as an explanation why we won't.

def get_string_encoding_layer(name, dataset, max_tokens=None):
  # Create a layer that integer encode strings
  string_encoding = layers.StringLookup(max_tokens=max_tokens, output_mode='int')

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them
  string_encoding.adapt(feature_ds)

  return string_encoding

# I'll explain a bit later why that output is not so good.
# But first problem - is strong divercity of values - we rather need some kind of
# integer encoding, where categorical data is treated independently.

string_encoding = get_string_encoding_layer('winery', train_ds)
string_encoding(train_sample_features['winery'])

def get_string_encoding_layer(name, dataset, max_tokens=None):
  # Create a layer that multi-hot encode strings
  string_encoding = layers.StringLookup(max_tokens=max_tokens, output_mode='multi_hot')

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them
  string_encoding.adapt(feature_ds)

  return string_encoding

# Multi-hot encoding doesn't look better.
# Just look at that wild-wide shape.
string_encoding = get_string_encoding_layer('winery', train_ds)
string_encoding(train_sample_features['winery'])

# We also can use INT output mode, and than use Normalization layer, or
# use Embedding layer later in the model.
# But we don't need any of this, since the hub layer works just fine.

"""APPLY ENCODING LAYERS AS THE INPUT LAYERS OF THE MODEL


"""

batch_size = 1024
label = 'points'
train_ds = df_to_dataset(train, label=label, batch_size=batch_size)
val_ds = df_to_dataset(val, label=label, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, label=label, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numerical features.
header = 'price'

normalization_layer = get_normalization_layer(header, train_ds)
numeric_input = tf.keras.Input(shape=(1,), name=header)
encoded_numeric_input = normalization_layer(numeric_input)

all_inputs.append(numeric_input)
encoded_features.append(encoded_numeric_input)

# Categorical features.
categorical_cols = ['country', 'province', 'variety']
max_tokens = 50

for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=max_tokens)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)

# I've thought - what if max_tokens don't pick up the most frequent data?
# That definetly would hurt our 'province' and 'variety' data.
# But, there is an answer in Keras documentation:

# If the vocabulary is capped in size, the most frequent
# tokens will be used to create the vocabulary and all others will be treated
# as out-of-vocabulary (OOV).

# So, we don't need to do any extra-steps.

# Text features.
headers = ['winery', 'title', 'description']

for header in headers:
  text_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = text_encoding
  encoded_text_col = encoding_layer(text_col)
  all_inputs.append(text_col)
  encoded_features.append(encoded_text_col)

encoded_features

"""MODEL"""

# We already converted all_input_layers -> all_encoding_layers.

# Let's create the separate layers of our model:

# Concatenete all_encoding_layers as an input layer.
all_features = tf.keras.layers.concatenate(encoded_features)
# Create a layer with 512 RELU hidden neurons.
dense_layer = tf.keras.layers.Dense(512, activation="relu")(all_features)
# Dropout will create some kind of noise - it's good for generalization of data.
dropout_layer = tf.keras.layers.Dropout(0.4)(dense_layer)
# One more layer of hidden neurons 
dense_layer_2 = tf.keras.layers.Dense(128, activation="relu")(dropout_layer)
# Output layer - RELU func is reccomended for scalar value output.
output = tf.keras.layers.Dense(1, activation="relu")(dense_layer_2)

# And model is done!
model = tf.keras.Model(all_inputs, output)

# Compile model - for loss function we use MSE, because we predict scalar value
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=tf.keras.metrics.MeanAbsoluteError())

# Use `rankdir='LR'` to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Train the model. 
history = model.fit(x=train_ds,
                    epochs=15,
                    validation_data=(val_ds)
                   )

# Let's look, how good our model. 
loss, mse = model.evaluate(test_ds)
print("MSE", mse)

# MSE 6.6 : It's not great, not bad. 

# Lets take the samples and plot predicted and test data.

SAMPLE_AMOUNT = 30

predicted_sample = model.predict(test_ds.take(1)) 
predicted_sample = predicted_sample[:SAMPLE_AMOUNT]
test_sample = test['points'].values[:SAMPLE_AMOUNT]

plt.plot(test_sample, color='red', label='test', linestyle='dashed', linewidth=2)
plt.plot(predicted_sample, color='blue', label='predicted')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Points')
plt.show()

# Predicted data follows some trends,
# but we can't call model really accurate.

# As I said - 
# for the sake of demonstration this model has been made unnecessarily complex.

# I propose next steps to improve this model:

# 1. Let's make it lighter.
# 1.1 Determine the dependence and influence of data on the value we're looking for.
# 1.2 Make some sort of graduation of features: [ important --> unimportant ].
# 1.3 Try to train model with most important feature, 2 features, 3 feat, etc. 
# 1.4 Pinpoint the border, where complexity of data become unnecessarily. 

# 2. Test hyperparameters.
# 2.1 Test model with different batch sizes, epochs, learning rate etc. 

# 3. Change model itself.
# 3.1 Add LSTM layer for natural language 'description' feature. 
#     It works better with contex-important data.
# 3.2 Try to make model bigger or smaller. Test accuracy and performance.

# 4. I don't see much point in messing with the   loss function,
#    but if being throught, we can change it and test the impact.