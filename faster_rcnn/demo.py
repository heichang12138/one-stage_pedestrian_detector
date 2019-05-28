import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import glob
import pprint

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import im_detect_rpn
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

def vis_detections(im, dets, thresh=0.5):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im[:,:,::-1], aspect='equal')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='g', linewidth=3)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.2f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='A demo of one-stage pedestrian detector')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', action='store_true')
    parser.add_argument('--net', dest='demo_net', choices=['VGGnet_test', 'MSnet_test'], default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path', required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    pprint.pprint(cfg)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    net = get_network(args.demo_net)
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print (' done.')
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))
    timer = Timer()
    for im_name in im_names:
        print 'Demo for {:s}'.format(im_name)
        im = cv2.imread(im_name)
        timer.tic()
        scores, boxes = im_detect_rpn(sess, net, im)
        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.5)
        dets = dets[keep, :]
        vis_detections(im, dets, thresh=0.5)