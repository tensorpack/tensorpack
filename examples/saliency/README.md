## Visualize Saliency Maps

Implement the Guided-ReLU visualization used in the paper:

* [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)

`saliency-maps.py` takes an image, and produce its saliency map by running a ResNet-50 and backprop its maximum
activations back to the input image space.
Similar techinques can be used to visualize the concept learned by each filter in the network.

Usage:
````bash
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzvf resnet_v1_50_2016_08_28.tar.gz
./saliency-maps.py cat.jpg
````

<p align="center"> <img src="./guided-relu-demo.jpg" width="800"> </p>

Left to right:
+ the original cat image
+ the magnitude in the saliency map
+ the magnitude blended with the original image
+ positive correlated pixels (keep original color)
+ negative correlated pixels (keep original color)
