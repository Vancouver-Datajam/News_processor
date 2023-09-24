# Project: News Processor | Vancouver DataJams 2023
# Date created: Sept 23, 2023 
# Python version: Python 3.9.13

## Data set used: https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset

import tensorflow as tf
import requests
import zipfile
import requests
import os
import time
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.models import load_model, Model
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import matplotlib.pyplot as plt
import glob
from functools import partial
import nltk

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        print("Couldn't set memory_growth")
        pass


def fix_random_seed(seed):
    """ Setting the random seed of various libraries """
    try:
        np.random.seed(seed)
    except NameError:
        print("Warning: Numpy is not imported. Setting the seed for Numpy failed.")
    try:
        tf.random.set_seed(seed)
    except NameError:
        print("Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed.")
    try:
        random.seed(seed)
    except NameError:
        print("Warning: random module is not imported. Setting the seed for random failed.")

# Fixing the random seed
random_seed=4321
fix_random_seed(random_seed)

print("TensorFlow version: {}".format(tf.__version__))

import pandas as pd
import requests
'''
With regard to man-made climate change, the following scores reflect the
sentiment of the tweet
2 is factual
1 is pro
0 is neutral
-1 is anti
'''
file_name = "twitter_sentiment_data.csv"
zip_file_name = "twitter.zip"
download_path = os.path.join('./../data', zip_file_name)

if not os.path.exists('data'):
        os.mkdir('data')

if not os.path.exists(os.path.join('./../data', file_name)):
  with zipfile.ZipFile(download_path, 'r') as zip_ref:
    extraction_directory = "./../data"  # Define your extraction directory
    os.makedirs(extraction_directory, exist_ok=True)
    zip_ref.extractall(extraction_directory)
    print(f"ZIP file contents extracted to '{extraction_directory}'")

dataframe = pd.read_csv(os.path.join('./../data','twitter_sentiment_data.csv'))
print(dataframe.head())
print(dataframe.keys())

'''
Cleaning up data - removing null or empty data.
Will need to change the column header depending on the dataset ("message")
'''
print("Before cleaning up: {}".format(dataframe.shape))
dataframe = dataframe[~dataframe["message"].isna()]
dataframe = dataframe[dataframe["message"].str.strip().str.len()>0]
print("After cleaning up: {}".format(dataframe.shape))

# Number of each sentiment in the dataset
dataframe["sentiment"].value_counts()

'''
Depending on what we want, we can decide on how we want to map
the sentiments to classses for training

For now, we can work off the assumption of the following
1 and 2 -> 1
0 and -1 -> 0

The final counts are printed out below
'''
dataframe["label"] = dataframe["sentiment"].map({1: 1, 2: 1, 0: 0, -1: 0})
dataframe["label"].value_counts()
print(dataframe.head())
print("\n")
print( dataframe["label"].value_counts())

# shuffle data
dataframe = dataframe.sample(frac=1.0, random_state=random_seed)

# split data into inputs and targets
inputs, labels = dataframe["message"], dataframe["label"]

'''
Performing the following
Lower case (nltk) - Turn "I am" to "i am"
Remove numbers (regex) - Turn "i am 24 years old" to "i am years old"
Remove stop words (nltk) - Turn "i go to the shop" to "i go shop"
Lemmatize (nltk) - Turn "i went to buy flowers" to "i go to buy flower"
'''

import nltk
nltk.download('averaged_perceptron_tagger', download_dir='nltk')
nltk.download('wordnet', download_dir='nltk')
nltk.download('stopwords', download_dir='nltk')
nltk.download('punkt', download_dir='nltk')
nltk.download('omw-1.4', download_dir='nltk')
nltk.data.path.append(os.path.abspath('nltk'))

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

rerun = False

# Define a lemmatizer (converts words to base form)
lemmatizer = WordNetLemmatizer()

# Define the English stopwords
EN_STOPWORDS = set(stopwords.words('english')) - {'not', 'no'}

# Code listing 9.2
def clean_text(doc):
    """ A function that cleans a given document (i.e. a text string)"""

    # Turn to lower case
    doc = doc.lower()
    # the shortened form n't is expanded to not
    doc = re.sub(pattern=r"\w+n\'t ", repl="not ", string=doc)
    # shortened forms like 'll 're 'd 've are removed as they don't add much value to this task
    doc = re.sub(r"(?:\'ll |\'re |\'d |\'ve )", " ", doc)
    # numbers are removed
    doc = re.sub(r"/d+","", doc)
    # break the text in to tokens (or words), while doing that ignore stopwords from the result
    # stopwords again do not add any value to the task
    tokens = [w for w in word_tokenize(doc) if w not in EN_STOPWORDS and w not in string.punctuation]

    # Here we lemmatize the words in the tokens
    # to lemmatize, we get the pos tag of each token and
    # if it is N (noun) or V (verb) we lemmatize, else
    # keep the original form
    pos_tags = nltk.pos_tag(tokens)
    clean_text = [
        lemmatizer.lemmatize(w, pos=p[0].lower()) \
        if p[0]=='N' or p[0]=='V' else w \
        for (w, p) in pos_tags
    ]

    # return the clean text
    return clean_text

# Run a sample
sample_doc = 'She sells seashells by the seashore.'
print("Before clean: {}".format(sample_doc))
print("After clean: {}".format(clean_text(sample_doc)))

if rerun or \
    not os.path.exists('sentiment_inputs.pkl') or \
    not os.path.exists('sentiment_labels.pkl'):
    # Apply the transformation to the full text
    # this is time consuming
    print("\nProcessing all the review data ... This can take some time (several minutes)")
    inputs = inputs.apply(lambda x: clean_text(x))
    print("\tDone")

    print("Saving the data")
    inputs.to_pickle('sentiment_inputs.pkl')
    labels.to_pickle('sentiment_labels.pkl')

else:
    # Load the data from the disk
    print("Data already found. If you want to rerun anyway, set rerun=True")
    inputs = pd.read_pickle( 'sentiment_inputs.pkl')
    labels = pd.read_pickle( 'sentiment_labels.pkl')

for actual, clean in zip(dataframe["message"].iloc[:5], inputs.iloc[:5]):
  print("Actual: {}".format(actual))
  print("Clean: {}".format(clean))
  print('\n')

# tweets and their labels
print(inputs.head(n=20))
print(labels.head(n=20))

def train_valid_test_split(inputs, labels, train_fraction=0.8):
    """ Splits a given dataset into three sets; training, validation and test """

    # Separate indices of negative and positive data points
    neg_indices = pd.Series(labels.loc[(labels==0)].index)
    pos_indices = pd.Series(labels.loc[(labels==1)].index)

    n_valid = int(min([len(neg_indices), len(pos_indices)]) * ((1-train_fraction)/2.0))
    n_test = n_valid

    neg_test_inds = neg_indices.sample(n=n_test, random_state=random_seed)
    neg_valid_inds = neg_indices.loc[~neg_indices.isin(neg_test_inds)].sample(n=n_test, random_state=random_seed)
    neg_train_inds = neg_indices.loc[~neg_indices.isin(neg_test_inds.tolist()+neg_valid_inds.tolist())]

    pos_test_inds = pos_indices.sample(n=n_test, random_state=random_seed)
    pos_valid_inds = pos_indices.loc[~pos_indices.isin(pos_test_inds)].sample(n=n_test, random_state=random_seed)
    pos_train_inds = pos_indices.loc[
        ~pos_indices.isin(pos_test_inds.tolist()+pos_valid_inds.tolist())
    ]

    tr_x = inputs.loc[neg_train_inds.tolist() + pos_train_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    tr_y = labels.loc[neg_train_inds.tolist() + pos_train_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    v_x = inputs.loc[neg_valid_inds.tolist() + pos_valid_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    v_y = labels.loc[neg_valid_inds.tolist() + pos_valid_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    ts_x = inputs.loc[neg_test_inds.tolist() + pos_test_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    ts_y = labels.loc[neg_test_inds.tolist() + pos_test_inds.tolist()].sample(frac=1.0, random_state=random_seed)

    print('Training data: {}'.format(len(tr_x)))
    print('Validation data: {}'.format(len(v_x)))
    print('Test data: {}'.format(len(ts_x)))

    return (tr_x, tr_y), (v_x, v_y), (ts_x, ts_y)

(tr_x, tr_y), (v_x, v_y), (ts_x, ts_y) = train_valid_test_split(inputs, labels)

print("Some sample targets")
print(tr_y.head(n=10))

'''
Analysis of vocabulary
'''
from collections import Counter
# Create a large list which contains all the words in all the reviews
data_list = [w for doc in tr_x for w in doc]

# Create a Counter object from that list
# Counter returns a dictionary, where key is a word and the value is the frequency
cnt = Counter(data_list)

# Convert the result to a pd.Series
freq_df = pd.Series(list(cnt.values()), index=list(cnt.keys())).sort_values(ascending=False)
# Print most common words
print(freq_df.head(n=10))

# Print summary statistics
print(freq_df.describe())

'''
Analysing thesequence length (number of words) of tweets
'''
# Create a pd.Series, which contain the sequence length for each review
seq_length_ser = tr_x.str.len()

# Get the median as well as summary statistics of the sequence length
print("\nSome summary statistics")
print("Median length: {}\n".format(seq_length_ser.median()))
seq_length_ser.describe()

print("\nComputing the statistics between the 10% and 90% quantiles (to ignore outliers)")
p_10 = seq_length_ser.quantile(0.1)
p_90 = seq_length_ser.quantile(0.9)

seq_length_ser[(seq_length_ser >= p_10) & (seq_length_ser < p_90)].describe(percentiles=[0.33, 0.66])

n_vocab = (freq_df >= 14).sum()
print("Using a vocabulary of size: {}".format(n_vocab))

'''
Transforming text to numbers
'''
from tensorflow.keras.preprocessing.text import Tokenizer

# Define a tokenizer that will convert words to IDs
# words that are less frequent will be replaced by 'unk'
tokenizer = Tokenizer(num_words=n_vocab, oov_token='unk', lower=False)

# Fit the tokenizer on the data
tokenizer.fit_on_texts(tr_x.tolist())

# Convert all of train/validation/test data to sequences of IDs
tr_x = tokenizer.texts_to_sequences(tr_x.tolist())
v_x = tokenizer.texts_to_sequences(v_x.tolist())
ts_x = tokenizer.texts_to_sequences(ts_x.tolist())

# Checking the attributes of the tokenizer
word = "rt"
wid = tokenizer.word_index[word]
print("The word id for \"{}\" is: {}".format(word, wid))
wid = 4
word = tokenizer.index_word[wid]
print("The word for id {} is: {}".format(wid, word))

# Convert words to IDs

# Vocabs not used get assigned id of 1 I think, so since they're not being
# used in the training, it doesn't actually matter
test_text = [
    ['rt', 'emptywheel', 'area', 'law', 'man', 'not', 'believe', 'climate', 'change', 'much', 'else', 'science', 'worry', 'cpd', 'report', 'not', 'scientifically', 'b…'],
    ['rt', 'seiclimate', 'think', 'climate', 'change', 'hoax', 'visit', 'norway', 'minister', 'say', 'climatehome', 'http', '//t.co/qs4zetknva', '“', 'seeing…'],
    ['’', 'build', 'man', '–', 'need', 'global', 'warming', '’', 'say', 'ivanka', '’', 'office', 'know', 'please', 'not', 'feel'],
    ['rt', 'trillburne', 'rejection', 'basic', 'observable', 'reality', 'birtherism', 'global', 'warm', 'denial', 'professor', 'read', 'atlanticâ€¦'],
    ['kenya', '’', 'climate', 'change', 'bill', 'aim', 'promote', 'low', 'carbon', 'growth', 'http', '//t.co/yiwqofn3ki'],
]

test_seq = tokenizer.texts_to_sequences(test_text)

for text, seq in zip(test_text, test_seq):
    print("Text: {}".format(text))
    print("Sequence: {}".format(seq))
    print("\n")

def get_tf_pipeline(text_seq, labels, batch_size=64, bucket_boundaries=[5,15], max_length=50, shuffle=False):
    """ Define a data pipeline that converts sequences to batches of data """

    # Concatenate the label and the input sequence so that we don't mess up the order when we shuffle
    data_seq = [[b]+a for a,b in zip(text_seq, labels) ]
    # Define the variable sequence dataset as a ragged tensor
    tf_data = tf.ragged.constant(data_seq)[:,:max_length]
    # Create a dataset out of the ragged tensor
    text_ds = tf.data.Dataset.from_tensor_slices(tf_data)

    text_ds = text_ds.filter(lambda x: tf.size(x)>1)
    # Bucketing the data
    # Bucketing assign each sequence to a bucket depending on the length
    # If you define bucket boundaries as [5, 15], then you get buckets,
    # [0, 5], [5, 15], [15,inf]
    bucket_fn = tf.data.experimental.bucket_by_sequence_length(
        lambda x: tf.cast(tf.shape(x)[0],'int32'),
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[batch_size,batch_size,batch_size],
        padded_shapes=None,
        padding_values=0,
        pad_to_bucket_boundary=False
    )

    # Apply bucketing
    text_ds = text_ds.map(lambda x: x).apply(bucket_fn)

    # Shuffle the data
    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10*batch_size)

    # Split the data to inputs and labels
    text_ds = text_ds.map(lambda x: (x[:,1:], x[:,0]))

    return text_ds

'''Testing the bucketing function '''
train_ds = get_tf_pipeline(tr_x, tr_y, shuffle=True)
valid_ds = get_tf_pipeline(v_x, v_y)

print("Some training data ...")
for x,y in train_ds.take(2):
    print("Input sequence shape: {}".format(x.shape))
    print(y)

print("\nSome validation data ...")
for x,y in valid_ds.take(2):
    print("Input sequence shape: {}".format(x.shape))
    print(y)

import tensorflow.keras.backend as K

K.clear_session()

class OnehotEncoder(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OnehotEncoder, self).__init__(**kwargs)
        self.depth = depth

    def build(self, input_shape):
        pass

    def call(self, inputs):

        inputs = tf.cast(inputs, 'int32')

        if len(inputs.shape) == 3:
            inputs = inputs[:,:,0]

        return tf.one_hot(inputs, depth=self.depth)


    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({'depth': self.depth})
        return config

# You will see the following error if you don't filter out all zero (empty) records from the dataset
# these records return a vector of all zeros which leads the LSTM layer to error out
# CUDNN_STATUS_BAD_PARAM
# in tensorflow/stream_executor/cuda/cuda_dnn.cc(1496):
# 'cudnnSetRNNDataDescriptor(
#     data_desc.get(), data_type,
#     layout,
#     max_seq_length, batch_size, data_size, seq_lengths_array, (void*)&padding_fill
# )'

# Code listing 9.5
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=(None,)),
    # Create a mask to mask out zero inputs
    tf.keras.layers.Masking(mask_value=0),
    # After creating the mask, convert inputs to onehot encoded inputs
    OnehotEncoder(depth=n_vocab),
    # Defining an LSTM layer
    tf.keras.layers.LSTM(128, return_state=False, return_sequences=False),
    # Defining a Dense layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

'''Checking the mask '''
inp = tf.expand_dims(tf.constant([[2,3,4,0,0], [2,4,6,12,0]], dtype='int32'),axis=-1)
# Create a mask to mask out zero inputs
mask_out = tf.keras.layers.Masking(mask_value=0)(inp)
print("Masking layer's mask")
print(mask_out._keras_mask)
# After creating the mask, convert inputs to onehot encoded inputs
onehot_out = OnehotEncoder(depth=10)(mask_out)
print("Onehot encoder layer's mask")
print(onehot_out._keras_mask)
# Defining an LSTM layer
lstm_out = tf.keras.layers.LSTM(24, return_state=False, return_sequences=False)(
    onehot_out, mask=onehot_out._keras_mask
)

print("Defining data pipelines")

# Using a batch size of 128
batch_size = 128

train_ds = get_tf_pipeline(tr_x, tr_y, batch_size=batch_size, shuffle=True)
valid_ds = get_tf_pipeline(v_x, v_y, batch_size=batch_size)
test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=batch_size)
print('\tDone...')

# There is a class imbalance in the data therefore we are defining a weight for negative inputs
neg_weight = (tr_y==1).sum()/(tr_y==0).sum()
print("Will be using a weight of {} for negative samples".format(neg_weight))

# Section 9.5

os.makedirs('eval', exist_ok=True)

# Logging the performance metrics to a CSV file
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('eval','1_sentiment_analysis.log'))

monitor_metric = 'val_loss'
mode = 'min'
print("Using metric={} and mode={} for EarlyStopping".format(monitor_metric, mode))

# Reduce LR callback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8
)

# EarlyStopping itself increases the memory requirement
# restore_best_weights will increase the memory req for large models
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor=monitor_metric, patience=6, mode=mode, restore_best_weights=False
)

# Train the model
t1 = time.time()

model.fit(train_ds, validation_data=valid_ds, epochs=10, class_weight={0:neg_weight, 1:1.0}, callbacks=[es_callback, lr_callback, csv_logger])
t2 = time.time()

print("It took {} seconds to complete the training".format(t2-t1))

os.makedirs('models', exist_ok=True)
tf.keras.models.save_model(model, os.path.join('models', '1_sentiment_analysis.h5'))

model.evaluate(test_ds)

# Section 9.6

import tensorflow.keras.backend as K

K.clear_session()

# Code listing 9.7
model = tf.keras.models.Sequential([

    # Adding an Embedding layer
    # You will see the following error if you don't filter out all zero (empty) records from the dataset
    # these records return a vector of all zeros which leads the LSTM layer to error out
    # CUDNN_STATUS_BAD_PARAM
    # in tensorflow/stream_executor/cuda/cuda_dnn.cc(1496):
    # 'cudnnSetRNNDataDescriptor(
    #     data_desc.get(), data_type,
    #     layout,
    #     max_seq_length, batch_size, data_size, seq_lengths_array, (void*)&padding_fill
    # )'
    tf.keras.layers.Embedding(input_dim=n_vocab+1, output_dim=128, mask_zero=True, input_shape=(None,)),
    # Defining an LSTM layer
    tf.keras.layers.LSTM(128, return_state=False, return_sequences=False),
    # Defining Dense layers
    tf.keras.layers.Dense(512, activation='relu'),
    # Defining a dropout layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Section 9.7

print("Defining data pipelines")
batch_size = 128
train_ds = get_tf_pipeline(tr_x, tr_y, batch_size=batch_size, shuffle=True)
valid_ds = get_tf_pipeline(v_x, v_y, batch_size=batch_size,)
test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=batch_size)
print('\tDone...')

os.makedirs('eval', exist_ok=True)

# Logging the performance metrics to a CSV file
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('eval','2_sentiment_analysis_embeddings.log'))

monitor_metric = 'val_loss'
mode = 'min' if 'loss' in monitor_metric else 'max'
print("Using metric={} and mode={} for EarlyStopping".format(monitor_metric, mode))

# Reduce LR callback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8
)

# EarlyStopping itself increases the memory requirement
# restore_best_weights will increase the memory req for large models
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor=monitor_metric, patience=6, mode=mode, restore_best_weights=False
)

t1 = time.time()

model.fit(train_ds, validation_data=valid_ds, epochs=10, class_weight={0:neg_weight, 1:1.0}, callbacks=[es_callback, lr_callback, csv_logger])
t2 = time.time()

print("It took {} seconds to complete the training".format(t2-t1))

os.makedirs('models', exist_ok=True)
tf.keras.models.save_model(model, os.path.join('models', '2_sentiment_analysis_embeddings.h5'))

test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=128)
model.evaluate(test_ds)

test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=128)

# Go through the test data and gather all examples
test_x = []
test_pred = []
test_y = []
for x, y in test_ds:
    test_x.append(x)
    test_pred.append(model.predict(x, verbose=0))
    test_y.append(y)

# Check the sizes
test_x = [doc for t in test_x for doc in t.numpy().tolist()]
print("X: {}".format(len(test_x)))
test_pred = tf.concat(test_pred, axis=0).numpy()
print("Pred: {}".format(test_pred.shape))
test_y = tf.concat(test_y, axis=0).numpy()
print("Y: {}".format(test_y.shape))

sorted_pred = np.argsort(test_pred.flatten())
min_pred = sorted_pred[:5]
max_pred = sorted_pred[-5:]

print("Most negative reviews\n")
print("="*50)
for i in min_pred:
    print(" ".join(tokenizer.sequences_to_texts([test_x[i]])), '\n')

print("\nMost positive reviews\n")
print("="*50)
for i in max_pred:
    print(" ".join(tokenizer.sequences_to_texts([test_x[i]])), '\n')