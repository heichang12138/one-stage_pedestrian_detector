import numpy as np
import numpy.random as npr
from scipy.misc import imresize

from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps, bbox_intersections
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, dontcare_areas, im_info, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):

    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0

    im_info = im_info[0]
    height, width = rpn_cls_score.shape[1:3]

    if cfg.SDS.SDS_ON:
        mask, mask_weights = _generate_mask(
                im_info, width, height, gt_boxes, dontcare_areas)
        rpn_mask = mask.reshape((1, height, width, 1))
        rpn_mask_weights = mask_weights.reshape((1, height, width, 1))

    else:
        rpn_mask = np.array([], dtype=np.float32)
        rpn_mask_weights = np.array([], dtype=np.float32)

    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    anchors = all_anchors[inds_inside, :]

    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # preclude dontcare areas
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x A
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float),
            np.ascontiguousarray(anchors, dtype=np.float)
        )
        intersecs_ = intersecs.sum(axis=0)
        labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if not cfg.RETINA.RETINA_ON:

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)

    rpn_labels = labels.reshape((1, height, width, A))
    rpn_bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
            rpn_mask, rpn_mask_weights

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _generate_mask(im_info, width, height, gt_boxes, dontcare_areas=None):
    """generate mask used in rpn._get_rpn_blobs (resize on the fly)"""
    mean_height = cfg.SDS.MEAN_HEIGHT * im_info[2]
    im_height, im_width = im_info[:2]

    mask = np.zeros((int(im_height), int(im_width)), dtype=np.float32)
    mask_weights = np.ones((int(im_height), int(im_width)), dtype=np.float32)

    for dontcare in dontcare_areas:
        x1 = int(min(max(round(dontcare[0]), 0), im_width-1))
        y1 = int(min(max(round(dontcare[1]), 0), im_height-1))
        x2 = int(min(max(round(dontcare[2]), 0), im_width-1))
        y2 = int(min(max(round(dontcare[3]), 0), im_height-1))
        mask[y1:y2, x1:x2] = 1
        mask_weights[y1:y2, x1:x2] = 0

    for gt in gt_boxes:
        x1 = int(min(max(round(gt[0]), 0), im_width-1))
        y1 = int(min(max(round(gt[1]), 0), im_height-1))
        x2 = int(min(max(round(gt[2]), 0), im_width-1))
        y2 = int(min(max(round(gt[3]), 0), im_height-1))
        h = y2 - y1
        mask[y1:y2, x1:x2] = 1
        mask_weights[y1:y2, x1:x2] = (1 + (h + 0.0) / mean_height)
    # default mode will turn int to uint8 and 1 will become 255
    # use mode='F' to keep things unchanged.
    mask = imresize(mask, (height, width), interp='nearest', mode='F')
    mask_weights = imresize(
            mask_weights, (height, width), interp='nearest', mode='F')

    return mask, mask_weights