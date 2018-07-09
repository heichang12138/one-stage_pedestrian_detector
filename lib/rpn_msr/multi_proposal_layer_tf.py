import numpy as np

from .generate_anchors import generate_anchors
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fast_rcnn.nms_wrapper import nms

def multi_proposal_layer(rpn_cls_prob_reshape_C5, rpn_cls_prob_reshape_C6,
                   rpn_bbox_pred_C5, rpn_bbox_pred_C6,
                   im_info, cfg_key,
                   multi_feat_stride = [16, 32],
                   anchor_scales_C5 = [8, 16, 32],
                   anchor_scales_C6 = [8, 16, 32]):

    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    im_info = im_info[0]
    rpn_blobs = [[],[]]
    for ci, (anchor_scales, _feat_stride, rpn_cls_prob_reshape, rpn_bbox_pred) in \
        enumerate(zip([anchor_scales_C5, anchor_scales_C6],
                      multi_feat_stride,
                      [rpn_cls_prob_reshape_C5,rpn_cls_prob_reshape_C6],
                      [rpn_bbox_pred_C5, rpn_bbox_pred_C6])):
        _anchors = generate_anchors(scales=np.array(anchor_scales))
        _num_anchors = _anchors.shape[0]

        height, width = rpn_cls_prob_reshape.shape[1:3]
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        A = _num_anchors
        K = shifts.shape[0]
        anchors = _anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:,:,:,:,1],
                            [1, height, width, _num_anchors])
        scores = scores.reshape((-1, 1))
        bbox_deltas = rpn_bbox_pred.reshape((-1, 4))

        proposals = bbox_transform_inv(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        rpn_blobs[ci].append(proposals)
        rpn_blobs[ci].append(scores)

    # stack them together and do another round nms
    proposals_out = np.concatenate([rpn_blobs[0][0], rpn_blobs[1][0]], axis=0)
    scores_out = np.concatenate([rpn_blobs[0][1], rpn_blobs[1][1]], axis=0)

    order = scores_out.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals_out = proposals_out[order, :]
    scores_out = scores_out[order]

    keep = nms(np.hstack((proposals_out, scores_out)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals_out = proposals_out[keep, :]
    scores_out = scores_out[keep]

    blob = np.hstack((proposals_out.astype(np.float32, copy=False),scores_out))

    return blob

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep