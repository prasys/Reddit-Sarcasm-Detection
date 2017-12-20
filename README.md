# Detecting Sarcasm Using LSTMs with Word-Embedding Features

To test an arbitrary string:
edit the `input_comments` list in `reddit_sarcasm_exec.py`

The feature trials were run using `reddit_sarcasm_{base, ellipsis, upper, exclamation}.py`

The Riloff metrics can be run using `reddit_sarcasm_metrics.ipynb` but the `scrape.ipynb` must be run first (You need a Twitter API key)

The validation set can be run using `reddit_sarcasm_eval.py`

The model can be trained again by using `reddit_sarcasm_{base, features}_large.py`

Dependencies:
* numpy
* scipy
* pandas
* TwitterAPI
* TensorFlow or Theano (Only tested with TensorFlow)
* keras
* sklearn
* h5py
