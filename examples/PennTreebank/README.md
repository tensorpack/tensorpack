
# LSTM language modeling on Penn Treebank dataset

This example is mainly to demonstrate:

1. How to train an RNN with persistent state between iterations.
2. How to use a TF reader pipeline instead of a DataFlow, for both training & inference.

It trains an language model on PTB dataset, basically an equivalent of the PTB example
in [tensorflow/models](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).
It has the same performance & speed as the original example as well.
Note that the data pipeline is completely copied from the tensorflow example.

To Train:
```
./PTB-LSTM.py
```


