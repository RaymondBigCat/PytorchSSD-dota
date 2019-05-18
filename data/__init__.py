# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .dota import DOTADetection, AnnotationTransform, detection_collate, DOTA_CLASSES
from .coco import COCODetection
from .data_augment import *
from .config import *
