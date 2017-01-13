
# Colorize

As creating a neural network for digit classification seems to be a bit outdated, we will create a fictional network that learns to colorize grayscale images. In this case-study, you will learn to do the following using TensorPack.

- Dataflow
    + create a basic dataflow containing images
    + debug you dataflow
    + add custom manipulation to your data such as converting to Lab-space
    + efficiently prefetch data
- Network
    + define a neural network architecture for regression
    + integrate summary functions of TensorFlow
- Training
    + create a training configuration
- Callbacks
    + write your own callback to export predicted images after each epoch

## Dataflow

The basic idea is to gather a huge amount of images, resizing them to the same size and extract the luminance channel after converting from RGB to Lab. For demonstration purposes, we will split the dataflow definition into single steps, though it might more efficient to combine some steps.


### Reading data
The first node in the dataflow is the image-reader. There are several ways to read a lot of images:

- use lmdb files you probably already used for the Caffe framework
- collect images from a specific directory
- read data from the ImageNet if you have already downloaded these images

We will use simply a directory which consists of many RGB images. This is as simple as:

````python
from tensorpack import *
import glob, os

imgs = glob.glob(os.path.join('/media/data/img', '*.jpg'))
ds = ImageFromFile(imgs, channel=3, shuffle=True)
ds = PrintData(ds, num=2) # only for debugging
````

Running this will give you:
````
[0112 18:59:47 @common.py:600] DataFlow Info:
datapoint 0<2 with 1 elements consists of
   dp 0: is ndarray of shape (1920, 2560, 3) with range [0.0000, 255.0000]
datapoint 1<2 with 1 elements consists of
   dp 0: is ndarray of shape (850, 1554, 3) with range [0.0000, 255.0000]
````

To actually get data you can add

````python
for dp in ds.get_data():
    print dp[0] # this is RGB data!
````

This iteration is used in an additional process. Of course, without the `print` statement. There is some other magic behind the scenes. The dataflow checks of the image is actually an RGB image with 3 channels and skip those gray-scale images.


### Manipulate incoming data
Now, training a network which is not fully convolutional requires inputs of fixed size. Let us add this to the dataflow.

````python
from tensorpack import *
import glob, os

imgs = glob.glob(os.path.join('/media/data/img', '*.jpg'))
ds = ImageFromFile(imgs, channel=3, shuffle=True)
ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
ds = PrintData(ds, num=2) # only for debugging
````

It's time to convert the rgb information into the Lab space. In python, you would to something like

````python
from skimage import color
rgb = get_my_image()
lab = color.rgb2lab(rgb)
````

using the `scikit-image` pip package.

We should add this to our dataflow:

````python
from tensorpack import *
import glob, os
from skimage import color

imgs = glob.glob(os.path.join('/media/data/img', '*.jpg'))
ds = ImageFromFile(imgs, channel=3, shuffle=True)
ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
ds = MapData(ds, lambda dp: [color.rgb2lab(dp[0])])
ds = PrintData(ds, num=2) # only for debugging
````

We can enhance this version by writing
````python
from tensorpack import *
import glob, os
from skimage import color

def get_data():
    augs = [imgaug.Resize((256, 256)),
            imgaug.MapImage(color.rgb2lab)]

    imgs = glob.glob(os.path.join('/media/data/img', '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds, 32)
    ds = PrefetchData(ds, 4) # use queue size 4
    return ds
````

But wait! The alert reader makes a critical observation! We need the L channel *only* and we should add the RGB image as ground-truth data. Let's fix that.

````python
from tensorpack import *
import glob, os
from skimage import color

def get_data():
    augs = [imgaug.Resize((256, 256))]
    augs2 = [imgaug.MapImage(color.rgb2lab)]

    imgs = glob.glob(os.path.join('/media/data/img', '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, augs)
    ds = MapData(ds, lambda dp: [dp[0], dp[0]]) # duplicate
    ds = AugmentImageComponent(ds, augs2)
    ds = MapData(ds, lambda dp: [dp[0][:, :, 0], dp[1]]) # get L channel from first entry
    ds = BatchData(ds, 32)
    ds = PrefetchData(ds, 4) # use queue size 4
    return ds
````

Here, we simply duplicate the rgb image and only apply the `image augmentors` to the first copy of the datapoint. The output when using `PrintData` should be like

````
datapoint 0<2 with 2 elements consists of
   dp 0: is ndarray of shape (256, 256) with range [0, 100.0000]
   dp 1: is ndarray of shape (256, 256, 3) with range [0, 221.6387]
datapoint 1<2 with 2 elements consists of
   dp 0: is ndarray of shape (256, 256) with range [0, 100.0000]
   dp 1: is ndarray of shape (256, 256, 3) with range [0, 249.6030]

````

Again, do not use `PrintData` in combination with `PrefetchData` because the prefetch will be done in another process.

Well, this is probably not the most efficient way to encode this process. But it clearly demonstrates how much flexibility the `dataflow` gives.

## Network

If you are surprised how far we already are, you will enjoy how easy it is to define a network model. The most simple model is probably:

````python
class Model(ModelDesc):

    def _get_input_vars(self):
        pass

    def _build_graph(self, input_vars):
        self.cost = 0
````

The framework expects:
- a definition of inputs in `_get_input_vars`
- a computation graph containing the actual network layers in `_build_graph`
- a member `self.cost` representing the loss function we would like to minimize.

### Define inputs
Our dataflow produces data which looks like `[(256, 256), (256, 256, 3)]`. The first entry is the luminance channel as input and the latter is the original RGB image with all three channels. So we will write

````python
def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 256, 256), 'luminance'),
                InputVar(tf.int32, (None, 256, 256, 3), 'rgb')]
````

This is pretty straight forward, isn't it? We defined the shapes of the input and spend each entry a name. This is very generous of us and will us help later to build an inference mechanism.

From now, the `input_vars` in `_build_graph(self, input_vars)` will have the shapes `[(256, 256), (256, 256, 3)]` because of the completed method `_get_input_vars`. We can therefore write

````python
class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 256, 256), 'luminance'),
                InputVar(tf.int32, (None, 256, 256, 3), 'rgb')]

    def _build_graph(self, input_vars):
        luminance, rgb = input_vars  # (None, 256, 256), (None, 256, 256, 3)
        self.cost = 0
````


### Define architecture
So all we need to do is to define a network layout 
$$f\colon \mathbb{R}^{b \times 256 \times 256} \to \mathbb{R}^{b \times 256 \times 256 \times 3}$$ mapping our input to a plausible rgb image.

The process of coming up with such a network architecture is usually a soup of experience, a lot of trials and much time laced with magic or simply chance, depending what you prefer. We will use an auto-encoder with a lot of convolutions to squeeze the information through a bottle-neck (encoder) and then upsample from a hopefully meaningful compact representation (decoder). 

Because we are fancy, we will use a U-net layout with skip-connections.

````python
NF = 64
with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            with argscope(Conv2D, kernel_shape=4, stride=2,
                          nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
                # encoder
                e1 = Conv2D('conv1', luminance, NF, nl=LeakyReLU)
                e2 = Conv2D('conv2', e1, NF * 2)
                e3 = Conv2D('conv3', e2, NF * 4)
                e4 = Conv2D('conv4', e3, NF * 8)
                e5 = Conv2D('conv5', e4, NF * 8)
                e6 = Conv2D('conv6', e5, NF * 8)
                e7 = Conv2D('conv7', e6, NF * 8)
                e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                # decoder
                e8 = Deconv2D('deconv1', e8, NF * 8)
                e8 = Dropout(e8)
                e8 = ConcatWith(e8, 3, e7)

                e7 = Deconv2D('deconv2', e8, NF * 8)
                e7 = Dropout(e7)
                e7 = ConcatWith(e7, 3, e6)

                e6 = Deconv2D('deconv3', e7, NF * 8)
                e6 = Dropout(e6)
                e6 = ConcatWith(e6, 3, e5)

                e5 = Deconv2D('deconv4', e6, NF * 8)
                e5 = Dropout(e5)
                e5 = ConcatWith(e5, 3, e4)

                e4 = Deconv2D('deconv5', e65, NF * 4)
                e4 = Dropout(e4)
                e4 = ConcatWith(e4, 3, e3)

                e3 = Deconv2D('deconv6', e4, NF * 2)
                e3 = Dropout(e3)
                e3 = ConcatWith(e3, 3, e2)

                e2 = Deconv2D('deconv7', e3, NF * 1)
                e2 = Dropout(e2)
                e2 = ConcatWith(e2, 3, e1)

                prediction = Deconv2D('prediction', e2, 3, nl=tf.tanh)
````

There are probably many better tutorials about defining your network model. And there are definitely [better models](../../examples/GAN/image2image.py). You should check them later. A good way to understand layers from this library is to play with those examples. 

It should be noted that you can write your models using [tfSlim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) which comes along architectures [architectures and pre-trained models](https://github.com/tensorflow/models/tree/master/slim/nets) for image classification. The library automatically handles regularization and batchnorm updates from tfSlim. And you can directly load these pre-trained checkpoints from state-of-the-art models in TensorPack. Isn't this cool?

The remaining part is a boring L2-loss function given by:

````python
self.cost = tf.nn.l2_loss(prediction - rgb, name="L2 loss")
````

### Pimp the TensorBoard output

It is a good idea to track the progress of your training session using TensorBoard. This library provides several functions to simplify the output of summaries and visualization of intermediate states.

The following two lines

````python
add_moving_summary(self.cost)
tf.summary.image('colorized', prediction, max_outputs=10)
````

add a plot of the costs from our loss function and add some intermediate results to the tab of "images" inside TensorBoard. The updates are then triggered after each epoch.

## Training

Let's summarize: We have a model and data. The missing piece which stitches these parts together is the training protocol. It is only a [configuration](../../tensorpack/train/config.py#L23-#L29)

For the dataflow, we already implemented `get_data` in the first part. Specifying the learning rate is done by

````python
lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True) 
````

This essentially creates a non-trainable variable with initial value `1e-4` and also track this value inside TensorBoard. Let's have a look at the entire code:

````python
def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
    return TrainConfig(
        dataflow=dataset,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([StatPrinter(), PeriodicCallback(ModelSaver(), 3)])]),
        model=Model(),
        step_per_epoch=dataset.size(),
        max_epoch=100,
    )
````

There is not really new stuff. The model was implemented, and `max_epoch` is set to 100. This means 100 runs over the entire dataset. The alert reader who almost already had gone to sleep makes some noise: "Where is `dataset.size()` coming from?" This values represents all images in one directory and is forwarded by all mappings. If you have 42 images in your directory, then this value is 42. Satisfied with this answer, the alert reader went out of the room. But he will miss the most interesting part: the callback section. We will cover this in the next section.


## Callbacks

Until this point, we spoke about all necessary part of deep learning pipelines which are common from GANs, image-recognition and embedding learning. But sometimes you want to add your own code. We will now add a functionality which will export some entries of the tensor `prediction`. Remember, this is the result of the decoder part in our network.

To not mess up the code, there is a plug-in mechanism with callbacks. Our callback looks like

````python
class OnlineExport(Callback):
    def __init__(self):
        pass

    def _setup_graph(self):
        pass

    def _trigger_epoch(self):
       pass
````

So it has 3 methods, although there are some more. TensorPack is really conservative regarding the computation graph. After the network is constructed and all callbacks are initialized the graph is finalized. So once you started training, there is no way of adding nodes to the graph, which we actually want to do for inference.

Let us fill in some parts

````python
class OnlineExport(Callback):
    def __init__(self):
        self.cc = 0
        self.example_input = color.rgb2lab(cv2.imread('myimage.jpg')[:, :, [2, 1, 0]])[:, :, 0] # read rgb image and extract luminance

    def _setup_graph(self):
        self.predictor = self.trainer.get_predict_func(['luminance'], ['prediction'])

    def _trigger_epoch(self):
        pass
````

Can you remember the method `_get_input_vars` in our model? We used the name `luminance` to identify one input. If not, this is the best time to go back in this text and read how to specify input variables for the network. In the deconvolution step there was also:

````python
prediction = Deconv2D('prediction', e2, 3, nl=tf.tanh) # name is 'prediction'
````

These two names allows to build the inference part of the network in
````python
inputs = ['luminance']
outputs = ['prediction']
self.trainer.get_predict_func(inputs, outputs)
````

This is very convenient because in the `_tigger_epoch` we can use:
````python
def _trigger_epoch(self):
        hopefully_cool_rgb = self.pred([self.example_input])[0]
````

to do inference. Together this looks like

````python
class OnlineExport(Callback):
    def __init__(self):
        self.cc = 0
        self.example_input = color.rgb2lab(cv2.imread('myimage.jpg')[:, :, [2, 1, 0]])[:, :, 0]

    def _setup_graph(self):
        inputs = ['luminance']
        outputs = ['prediction']
        self.trainer.get_predict_func(inputs, outputs)

    def _trigger_epoch(self):
        hopefully_cool_rgb = self.pred([self.example_input])[0]
        cv2.imwrite("export%04i.jpg" % self.cc, hopefully_cool_rgb)
        self.cc += 1
````

Do not forget to add `OnlineExport` to you callbacks in the train-configuration.
