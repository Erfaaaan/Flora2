import logging
import torch
from collections import OrderedDict
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)

from roboflow import Roboflow

from matplotlib import pyplot as plt
from PIL import Image
import json


def li_of_flowers():
  total_flower_list = [
  'Acaulis',
  'Alstroemeria',
  'Artichoke',
  'Bommaji',
  'Calla',
  'Carnation',
  'Chamnahree',
  'Cosmos',
  'Curcuma',
  'Dalnancho',
  'Dandelion',
  'Dongja',
  'Doraji',
  'Echinacea',
  'Englishdaisy',
  'FranceGukhwa',
  'Fritillaria',
  'Geuknakjo',
  'Geumeocho',
  'Geumggyeguk',
  'Geumjanhwa',
  'Geummaehwa',
  'GgangGgangyee',
  'Gloriosa',
  'Guemgangchorong',
  'Gyeongyeopduran',
  'Halmi',
  'Jjille',
  'Lentenrose',
  'Marigold',
  'Minariajaebi',
  'Mugunghwa',
  'Mulmangcho',
  'Muskari',
  'Nigella',
  'NorangGgotchangpo',
  'Norugwi',
  'Poinsettia',
  'Suseonhwa',
  'Suyeompaeraengi',
  'Sweetpea',
  'Yongwang',
  'alpine_sea_holly',
  'anthurium',
  'azalea',
  'ball_moss',
  'barbeton_daisy',
  'bearded_iris',
  'bee_balm',
  'bishop_of_llandaff',
  'black_eyed_susan',
  'blackberry_lily',
  'blanket_flower',
  'bolero_deep_blue',
  'bougainvillea',
  'bromelia',
  'californian_poppy',
  'camellia',
  'canna_lily',
  'canterbury_bells',
  'cape_flower',
  'cautleya_spicata',
  'clematis',
  'columbine',
  'common_dandelion',
  'corn_poppy',
  'cyclamen',
  'desert-rose',
  'foxglove',
  'frangipani',
  'garden_phlox',
  'gaura',
  'gazania',
  'geranium',
  'globe_thistle',
  'great_masterwort',
  'hibiscus',
  'hippeastrum',
  'japanese_anemone',
  'lotus',
  'magnolia',
  'mallow',
  'mexican_petunia',
  'monkshood',
  'morning_glory',
  'orange_dahlia',
  'osteospermum',
  'passion_flower',
  'petunia',
  'pincushion_flower',
  'pink-yellow_dahlia',
  'pink_primrose',
  'primula',
  'prince_of_wales_feathers',
  'red_ginger',
  'rose',
  'ruby-lipped_cattleya',
  'silverbush',
  'spear_thistle',
  'spring_crocus',
  'sunflower',
  'sword_lily',
  'thorn_apple',
  'toad_lily',
  'tree_mallow',
  'trumpet_creeper',
  'water_lily'
  ]
  return total_flower_list




def counting_flowers(im):

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = os.path.join("model_final_15.pth")
  #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 10000 #We found that with a patience of 500, training will early stop before 10,000 iterations
  cfg.SOLVER.STEPS = []
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 108 # 106 letters plus one super class
  cfg.TEST.EVAL_PERIOD = 0 # Increase this number if you want to monitor validation performance during training

  PATIENCE = 500 #Early stopping will occur after N iterations of no imporovement in total_loss

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  cfg.MODEL.WEIGHTS = os.path.join("model_final_15.pth")  # path to the model we just trained

  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set a custom testing threshold
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                  scale=0.5,
                  instance_mode=ColorMode.IMAGE_BW
  )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  im = Image.fromarray(out.get_image()[:, :, ::-1])
  x=outputs["instances"].pred_classes
  list_of_flowers = x.tolist()
  dict_of_flowers = {}
  total_flower_list = li_of_flowers()
  j = 0
  for i in list_of_flowers:
    list_of_flowers[j] = total_flower_list[i-1]
    j += 1
  for i in range(len(list_of_flowers)):
    dict_of_flowers[list_of_flowers[i]] = list_of_flowers.count(list_of_flowers[i])

  json_data = json.dumps(dict_of_flowers)

  return im ,json_data
