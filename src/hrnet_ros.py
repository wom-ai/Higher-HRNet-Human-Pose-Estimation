#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import yaml
import cv2
import numpy as np

import _init_paths
import models
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from utils.transforms import get_final_preds
from utils.vis import show_image
import rospy
import torch
import torchvision
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import MultiArrayDimension

this_dir = os.path.dirname(__file__)

def OnImage(data, args):
  img_pub = args[0]
  info_pub = args[1]
  
  bridge = CvBridge()
  cv_image = bridge.imgmsg_to_cv2(data,"bgr8")
  transforms = torchvision.transforms.Compose(
      [
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )
      ]
  )
  image = transforms(cv_image)
  image = image.unsqueeze(0).cuda()

  with torch.no_grad():
    base_size = (640,640)
    outputs, heatmaps, tags = get_multi_stage_outputs(cfg, model, image, cfg['TEST']['FLIP_TEST'],cfg['TEST']['PROJECT2IMAGE'],base_size)
    parser = HeatmapParser(cfg)
    tags = tags[0].unsqueeze(4)
    grouped, scores = parser.parse(heatmaps[0], tags, cfg['TEST']['ADJUST'], cfg['TEST']['REFINE'])
    
  num_people = len(grouped[0])
  
  for i in reversed(range(num_people)):
    if scores[i] < 0.2:
      scores.pop(i)
      grouped[0] = np.delete(grouped[0],i,0)

  final_image = show_image(cv_image, grouped[0])

  #show image
  cv2.imshow("image",final_image)
  cv2.waitKey(3)

  #publish image
  final_image = bridge.cv2_to_imgmsg(final_image,"bgr8")
  img_pub.publish(final_image)
  
  if len(scores)==0:
    keypoints = []
  else:
    keypoints = np.delete(grouped[0],[2,3],2).flatten().tolist()

  #publish image info
  img_info = Int16MultiArray()
  
  img_info.layout.dim = []
  for i in range(3):
    img_info.layout.dim.append(MultiArrayDimension())
  img_info.layout.dim[0].label = "people"
  img_info.layout.dim[0].size = len(scores)
  img_info.layout.dim[0].stride = 2*17
  img_info.layout.dim[1].label = "keypoints"
  img_info.layout.dim[1].size = 17
  img_info.layout.dim[1].stride = 2
  img_info.layout.dim[2].label = "xy"
  img_info.layout.dim[2].size = 2
  img_info.layout.dim[2].stride = 1
 
  img_info.data = keypoints

  info_pub.publish(img_info)


def load_config(config_file):
  with open(config_file, 'r') as stream:
    try:
      return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
      print(exc)

def main():
  global cfg
  cfg_file = os.path.join(this_dir,'w32_512_adam_lr1e-3.yaml')
  cfg = load_config(cfg_file)
  
  global model
  model = eval('models.'+cfg['MODEL']['NAME']+'.get_pose_net')(
      cfg, is_train=False
  ).cuda()
  model.eval()
  model_state_file = os.path.join(this_dir,"../models/pose_higher_hrnet_w32_512.pth")
  model.load_state_dict(torch.load(model_state_file),strict=True)

if __name__ == '__main__':
  rospy.init_node("hrnet_pose")
  
  main()

  img_pub = rospy.Publisher('/hrnet_image',Image,queue_size=1)
  info_pub = rospy.Publisher('/hrnet_info',Int16MultiArray,queue_size=1)

  rospy.Subscriber("/rrbot/camera1/image_raw",Image,OnImage, [img_pub,info_pub])

  rospy.spin()
