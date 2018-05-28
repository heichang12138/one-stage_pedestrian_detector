import numpy as np
from ..fast_rcnn.config import cfg

def prepare_roidb(imdb):
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = cfg.IMAGE_WIDTH
        roidb[i]['height'] = cfg.IMAGE_HEIGHT

def add_bbox_regression_targets(roidb):
    assert len(roidb) > 0
    num_images = len(roidb)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (cfg.NCLASSES, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (cfg.NCLASSES, 1))
    else:
        pass

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0) # ignore bg class

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, cfg.NCLASSES):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    else:
        print "NOT normalizing targets"

    return means.ravel(), stds.ravel()