import argparse
import pprint
import numpy as np
import sys

from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg, cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network


def parse_args():
    parser = argparse.ArgumentParser(description='Train one-stage pedestrian detector')
    parser.add_argument('--weights', dest='pretrained_model', required=True, type=str)
    parser.add_argument('--network', dest='network_name', choices=['VGGnet_train', 'MSnet_train'], default='VGGnet_train')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', choices=['caltech_train'], default='caltech_train', type=str)
    parser.add_argument('--restore', dest='restore',action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)
    print('total samples: {:d}'.format(len(roidb)))
    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    print 'Logs will be saved to `{:s}`'.format(log_dir)
    print 'use device /gpu:0' # always use gpu 0
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)
    train_net(network, imdb, roidb, output_dir=output_dir, log_dir=log_dir,
              pretrained_model=args.pretrained_model, restore=args.restore)