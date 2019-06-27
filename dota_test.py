from __future__ import print_function

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #设置GPU1可见
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc,DOTA_500, DOTAroot, DOTADetection, DOTA_CLASSES
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

"""
parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='DOTA',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=64,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
# resume_time
parser.add_argument('--resume_time', default='2019-05-20-21:48',
                    type=str, help='resume time for retraining')

parser.add_argument('-max', '--max_epoch', default=400,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=10,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
#parser.add_argument('--save_folder', default='/home/buaab622/project/PytorchSSD-dota/weights/DOTAweights',help='Location to save checkpoint models')
parser.add_argument('--date', default='1213')
parser.add_argument('--save_frequency', default=20)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency', default=20)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
                    
# test or train
parser.add_argument('--test_mode', default=False, type=bool, help='switch to test_mode')
args = parser.parse_args()
"""
# set dataset(val or test, not train)
train_sets = 'subset_field_500_GSC' # A subset of DOTA for training
train_list = 'field_train' # dataset type
val_sets = 'subset_field_500_val_GSC' # A subset of DOTA for testing
val_list = 'field_val' # dataset type
cfg = DOTA_500

#save_date = time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time()))
# save_folder = os.path.join(args.save_folder, args.version + '_' + args.dataset +'_'+ args.size,save_date)
save_root = '/home/buaab622/project/PytorchSSD-dota/weights/DOTAweights/RFB_vgg_DOTA_300'
save_time = '2019-05-21-00:19'
save_folder = os.path.join(save_root, save_time)
log_file_path = save_folder + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
# save_folder = os.path.join(args.save_folder, args.version + '_' + args.size, args.date)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

# Build net
from models.RFB_Net_vgg import build_net
    
# Some Image Parameters
rgb_std = (1, 1, 1)
img_dim = 300
rgb_means = (104, 117, 123)
p = 0.6

# num_classes
num_classes = len(DOTA_CLASSES)

# Some Hyper-Parameters
batch_size = 64
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
net = build_net(img_dim, num_classes)
print(net)

# load saved weight
# resume_net_path:权重文件的路径
resume_epoch = 300
resume_net_path = os.path.join(save_folder, 'RFB_vgg_DOTA'+'_epoches_'+str(resume_epoch) + '.pth')
    
print('Loading resume network', resume_net_path)
state_dict = torch.load(resume_net_path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)

# Build utils for training
ngpu = 2
if ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))

cuda = True
if cuda:
    net.cuda()
    cudnn.benchmark = True

detector = Detect(num_classes, 0, cfg)

#optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)

# dataset
# DOTA
print('Loading Dataset...')

testset = DOTADetection(DOTAroot, 
                        val_sets, 
                        None, 
                        AnnotationTransform(),
                        dataset_name = val_list)
train_dataset = DOTADetection(DOTAroot,
                              train_sets,
                              preproc(img_dim, rgb_means, rgb_std, p), 
                              AnnotationTransform(), 
                              dataset_name = train_list)

# test net
def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = len(DOTA_CLASSES)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
    return APs, mAP

# run test
def test():
    net.eval()
    top_k = 300
    APs, mAP = test_net(test_save_dir, net, detector, cuda, testset,
                        BaseTransform(img_dim, rgb_means, rgb_std, (2, 0, 1)),
                        top_k, thresh=0.01)
    APs = [str(num) for num in APs]
    mAP = str(mAP)
    print('mAP = ',mAP)
    
if __name__ == '__main__':
    test()