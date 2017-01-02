Guided ReLU
===========

This is just taken from a side-note within:

**Striving for Simplicity: The All Convolutional Net**
Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller

To reproduce you need to download the official ResNet-50 model from tfSlim:
````bash
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzvf resnet_v1_50_2016_08_28.tar.gz
````

And download the [mean-file](https://drive.google.com/open?id=0B7gnqAAAoxJkOHIySmRzME9LZnc).

Running
````bash
python extract_saliency.py
````

will give you the compute saliency using guided ReLU:

absolute saliency
<p align="center"> <img src="./_abs_saliency.jpg" width="400"> </p>
positive saliency
<p align="center"> <img src="./_pos_saliency.jpg" width="400"> </p>
negative saliency
<p align="center"> <img src="./_neg_saliency.jpg" width="400"> </p>

More visualizations are

Highlight image-parts with high saliency:
<p align="center"> <img src="./_heatmap.jpg" width="400"> </p>

Convert any intensity information to heatmap:
<p align="center"> <img src="./_intensity.jpg" width="400"> </p>
