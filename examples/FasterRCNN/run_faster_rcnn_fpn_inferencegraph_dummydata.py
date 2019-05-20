import os
import time
import sys
import numpy as np
import tensorflow as tf
import argparse
import cv2

#from PIL import Image
from google.protobuf import text_format
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline 

from config import finalize_configs, config as cfg
from coco import COCODetection
from common import (
    DataFromListOfDict, CustomResize, filter_boxes_inside_shape,
    box_to_point8, point8_to_box, segmentation_to_mask)
from tensorpack.dataflow import (
    imgaug, TestDataSpeed,
    PrefetchDataZMQ, MultiProcessMapDataZMQ, MultiThreadMapData,
    MapDataComponent, DataFromList)
#from data import (
#    get_train_dataflow, get_eval_dataflow,
#    get_all_anchors, get_all_anchors_fpn)

#os.environ["KMP_BLOCKTIME"] = "0"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
#os.environ["OMP_NUM_THREADS"] = "28"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_eval_dataflow():
    finalize_configs(is_training=False)
    imgs = COCODetection.load_many(args.image_dir, args.data_val, add_gt=False)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id'])

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def run_inference(sess, image_tensor, detection_boxes,detection_probs,detection_labels,image_np_expanded, i):
       
  #image_np_expanded=np.random.rand(800, 1202, 3).astype(np.uint8)

  if not args.timeline: 
    (boxes, probs, labels) = sess.run([detection_boxes, detection_probs, detection_labels],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata ) 
  else:
    frozen_model_trace_path = args.main_dir + "trace/" + file_name + "/"
    ensure_dir(frozen_model_trace_path)
    frozen_model_trace_path = frozen_model_trace_path + "timeline_frcnnfpn_{}_dummy.json".format(i)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    (boxes, probs, labels) = sess.run([detection_boxes, detection_probs, detection_labels],feed_dict = {image_tensor : image_np_expanded},options=run_options, run_metadata=run_metadata) 
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    with gfile.Open(frozen_model_trace_path, 'w') as trace_file:
      trace_file.write(trace.generate_chrome_trace_format(show_memory=False))   
  


def DetectOneImageModelFuncReadfromFrozenGraph(input_image_np=None):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = 20
  config.inter_op_parallelism_threads = 1

  with tf.Session(config=config) as sess:
    file_name = args.model_name
    file_path= args.main_dir + file_name

    print("***************************************************") 
    print("Loading and inferencing model: {}".format(file_path))
    print("***************************************************")     
    with tf.gfile.FastGFile(file_path,'rb') as f:  # Load pb as graphdef
      graphdef = tf.GraphDef() 
      graphdef.ParseFromString(f.read()) 
      sess.graph.as_default()  
      tf.import_graph_def(graphdef, name='')

      # Definite input and output Tensors for detection_graph
      image_tensor = graph.get_tensor_by_name('image:0')  
      detection_boxes = graph.get_tensor_by_name('final_boxes:0')
      detection_probs = graph.get_tensor_by_name('final_probs:0')
      detection_labels = graph.get_tensor_by_name('final_labels:0')
      
      #preprocess images
      reshapedimg = []
      if(args.image_dir):
        df = get_eval_dataflow()
        df.reset_state()        
        for img, img_id in df.get_data():        
          orig_shape = img.shape[:2]
          resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
          resized_img = resizer.augment(img)
          #scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
          reshapedimg.append(resized_img)
          print("Preprocesing image", len(reshapedimg),  "and adding to input image list")
      ## Add data
      tf.global_variables_initializer()    
     
      ## INFERENCE Config
      # Choose type of inference 1. single or 2.average
      #inference_type = "single"
      inference_type = "average"
      # Do actual inference
      if inference_type == "single":
        (boxes, probs, labels) = sess.run([detection_boxes, detection_probs, detection_labels],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata )
        #for _ in range(500):
        #  (boxes, probs, labels) = sess.run([detection_boxes, detection_probs, detection_labels],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata )          
      else :      
        i = 0
        avg=0
        if(args.image_count): 
          print("Input: Dummy data")       
          for _ in range(args.image_count):  
            # input dummy data          
            image_np_expanded=np.random.rand(800, 1202, 3).astype(np.uint8)
            i+=1
            start_time = time.time()  
            run_inference(sess, image_tensor, detection_boxes,detection_probs,detection_labels,image_np_expanded, i)  
            print("current inference time: {} ".format (time.time() - start_time)) 
            if(i>1):
              avg+=(time.time()-start_time)            
              if(i == args.image_count):
                print('Average inference time: %.3f sec'%(float(avg)/float(i-1)))
        elif(args.image_dir):
          print("Input: Real data")
          args.image_count = len(reshapedimg)
          for img in reshapedimg:
            image_np_expanded=img
            i+=1
            start_time = time.time()  
            run_inference(sess, image_tensor, detection_boxes,detection_probs,detection_labels,image_np_expanded, i)  
            elapsed_time= (time.time() - start_time)
            print("current inference {} time: {} ".format(i, elapsed_time))
        
            if(i>1):
              avg+=(time.time()-start_time)            
              if(i == args.image_count):
                print('Average inference time: %.3f sec'%(float(avg)/float(i-1)))
        else:
          print("please specify image_count(for dummy data) or image_dir(containing real data \n")
      # return some thing
      return True #(boxes, probs, labels)

if __name__ == '__main__':
  main_dir_path = os.path.dirname(os.path.realpath(__file__)) + "/temp/built_graph/"
  model_name = "Fasterrcnnfpn_graph_def_freezed.pb"
  parser = argparse.ArgumentParser()
  parser.add_argument('--main_dir', help='main directory containing temp folder', default=main_dir_path)
  parser.add_argument('--model_name', help='name of the directory', default=model_name)
  parser.add_argument('--image_count', type=int, help='number of input image/loop count')
  parser.add_argument('--image_dir', help='directory number of input images')
  parser.add_argument('--data_val', help='the value of dataset', default='val2017')

  parser.add_argument('--timeline', action='store_true', help='fetch timeline for pb file.')
  parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')

  args = parser.parse_args()

  with tf.Graph().as_default() as graph:
    output=DetectOneImageModelFuncReadfromFrozenGraph()