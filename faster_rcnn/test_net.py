import sys
import os
import argparse
import pprint
import tensorflow as tf

from lib.fast_rcnn.test import test_net
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate one-stage pedestrian detector')
    parser.add_argument('--weights', dest='model', required=True, type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', choices=['caltech_test', 'caltech_train'], default='caltech_test', type=str)
    parser.add_argument('--network', dest='network_name', choices=['VGGnet_test', 'MSnet_test'], default='VGGnet_test')
    parser.add_argument('--vis', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    pprint.pprint(cfg)
    weights_filename = os.path.splitext(os.path.basename(args.model))[0]
    imdb = get_imdb(args.imdb_name)
    print 'Use device /gpu:0'
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)
    test_net(sess, network, imdb, weights_filename, vis=args.vis, thresh=0.05)