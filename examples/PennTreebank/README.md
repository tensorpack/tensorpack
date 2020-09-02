
# LSTM language modeling on Penn Treebank dataset

This example is mainly to demonstrate:

1. How to train an RNN with persistent state between iterations. Here it simply manages the state inside the graph.
2. How to use a TF reader pipeline instead of a DataFlow, for both training & inference.

It trains an language model on PTB dataset, and reimplements an equivalent of the PTB example
in [tensorflow/models](https://github.com/tensorflow/models/blob/v1.13.0/tutorials/rnn/ptb/ptb_word_lm.py)
with its "medium" config.
It has the same performance as the original example as well.

Note that the input data pipeline is completely copied from the tensorflow example.

To Train:
```
./PTB-LSTM.py
```


