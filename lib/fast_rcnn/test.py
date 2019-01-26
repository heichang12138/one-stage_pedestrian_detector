import numpy as np
import cv2
import cPickle
import os
import matplotlib.pyplot as plt

from .config import cfg, get_output_dir
from ..utils.timer import Timer
from ..utils.cython_nms import nms
from ..utils.blob import im_list_to_blob
from ..fast_rcnn.bbox_transform import bbox_transform_inv


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    cls_score, cls_prob, bbox_pred, rois = \
        sess.run([net.get_output('rpn_cls_score'), net.get_output('rpn_cls_prob'), net.get_output('rpn_bbox_pred'),net.get_output('rpn_rois')],\
                 feed_dict=feed_dict)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes = rois[:, 1:5] / im_scales[0]
    scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

def im_detect_rpn(sess, net, im, boxes=None):
    blobs, im_scales = _get_blobs(im, boxes)
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    rois = sess.run(net.get_output('rois'), feed_dict=feed_dict)
    boxes = rois[:,:4]/im_scales[0]
    scores = rois[:,-1]
    # for compatible with origial version
#    boxes = np.concatenate((boxes,boxes),axis=1)
#    scores = np.concatenate((scores[:,np.newaxis],scores[:,np.newaxis]),axis=1)
    return scores,boxes
        
def vis_detections(im, class_name, dets, thresh=0.05):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt 
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4] 
        score = dets[i, -1] 
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.gca().text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))

def test_net(sess, net, imdb, weights_filename , max_per_image=300, thresh=0.05, vis=False):
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in xrange(num_images):
        # filter out any ground truth boxes
        box_proposals = None
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect_rpn(sess, net, im, box_proposals)
        detect_time = _t['im_detect'].toc(average=False)

        cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        all_boxes[1][i] = cls_dets
        if vis:
            image = im[:, :, (2, 1, 0)] 
            plt.cla()
            plt.imshow(image)
            vis_detections(image, 'ped', cls_dets)
            plt.show()

        print 'im_detect: {:d}/{:d} {:.3f}s' .format(i + 1, num_images, detect_time)


    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)