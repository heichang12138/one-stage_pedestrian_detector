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
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test a RPN network')
    parser.add_argument('--weights', dest='model',
                        help='model to test', default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',help='dataset to test',
                        default='caltech_test', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='VGGnet_test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]

    imdb = get_imdb(args.imdb_name)
    print 'Use device /gpu:0'

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)

    test_net(sess, network, imdb, weights_filename, vis=False, thresh=0.05)
