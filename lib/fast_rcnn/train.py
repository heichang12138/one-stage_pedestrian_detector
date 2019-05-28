import numpy as np
import os
import tensorflow as tf
import cv2

from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..roi_data_layer import roidb as rdl_roidb
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, logdir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds \
                    = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(
                logdir=logdir, graph=tf.get_default_graph(), flush_secs=5)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred') and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        """A simple graph for write image summary"""
        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        from tensorflow.python.ops import gen_logging_ops
        from tensorflow.python.framework import ops as _ops
        log_image = gen_logging_ops._image_summary(
            log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        return log_image, log_image_data, log_image_name


    def train_model(self, sess, max_iters, restore=False):
        data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        if cfg.MULTISCALE.MS_ON:
            loss, rpn_cross_entropy, rpn_loss_box, rpn_loss_mask = self.net.build_ms_loss()
        else:
            loss, rpn_cross_entropy, rpn_loss_box, rpn_loss_mask = self.net.build_loss()

        global_step = tf.Variable(0, trainable=False)
        # scalar summary
        if cfg.SDS.SDS_ON:
            tf.summary.scalar('rpn_loss_mask', rpn_loss_mask)
        tf.summary.scalar('rpn_rgs_loss', rpn_loss_box)
        tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy)
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()
        # image writer
        # NOTE: this image is independent to summary_op
        log_image, log_image_data, log_image_name = self.build_image_summary()
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        opt = tf.train.MomentumOptimizer(lr, momentum)
        tvars = tf.trainable_variables()
        grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
            
        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load vgg16
        if self.pretrained_model is not None and not restore:
            print ('Loading pretrained model weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        # resuming a trainer
        if restore:
            ckpt = tf.train.get_checkpoint_state(self.pretrained_model)
            print(self.output_dir)
            print 'Restoring from {}...'.format(ckpt.model_checkpoint_path),
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
            restore_iter = int(stem.split('_')[-1])
            sess.run(global_step.assign(restore_iter))
            print 'done'

        last_snapshot_iter = -1
        timer = Timer()

        for iter in range(restore_iter, max_iters):
            timer.tic()
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
            blobs = data_layer.forward()

            feed_dict={
                self.net.data: blobs['data'],
                self.net.im_info: blobs['im_info'],
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: blobs['gt_boxes'],
                self.net.dontcare_areas: blobs['dontcare_areas']
            }
                
            res_fetches = [self.net.get_output('rpn_rois')]

            fetch_list = [rpn_cross_entropy,
                          rpn_loss_box,
                          summary_op,
                          train_op] + res_fetches

            fetch_list += [rpn_loss_mask]

            rpn_loss_cls_value, rpn_loss_box_value, summary_str, _, rois, \
            rpn_loss_mask_value = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            _diff_time = timer.toc(average=False)

            # image summary
            if (iter) % cfg.TRAIN.LOG_IMAGE_ITERS == 0:
                # plus mean
                ori_im = np.squeeze(blobs['data']) + cfg.PIXEL_MEANS
                ori_im = ori_im.astype(dtype=np.uint8, copy=False)
                ori_im = _draw_gt_to_image(ori_im, blobs['gt_boxes'])
                ori_im = _draw_dontcare_to_image(ori_im, blobs['dontcare_areas'])
                res = _wrapper(rois,threshold=0.5)
                image = cv2.cvtColor(_draw_boxes_to_image(ori_im, res), cv2.COLOR_BGR2RGB)
                log_image_name_str = ('%06d_' % iter ) + blobs['im_name']
                log_image_summary_op = \
                    sess.run(log_image, \
                             feed_dict={log_image_name: log_image_name_str,\
                                        log_image_data: image})
                self.writer.add_summary(log_image_summary_op, global_step=global_step.eval())

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                total_loss = rpn_loss_cls_value + rpn_loss_box_value + rpn_loss_mask_value
                print 'iter: %d / %d,'%(iter, max_iters),
                print 'total loss: %.4f,'%(total_loss),
                print 'cls loss: %.4f,'%(rpn_loss_cls_value),
                print 'box loss: %.4f,'%(rpn_loss_box_value),
                if cfg.SDS.SDS_ON:
                    print 'mask loss: %.4f,'%(rpn_loss_mask_value),
                print 'lr: %f'%( lr.eval())
                print 'speed: {:.3f}s / iter'.format(_diff_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def _wrapper(rois, threshold, class_sets=None):
    num_class = cfg.NCLASSES
    class_sets = ['class_' + str(i) for i in range(0, num_class)] if class_sets is None else class_sets
    res = []
    for ind, cls in enumerate(class_sets[1:]):
        dets = rois
        dets = dets[np.where(dets[:, 4] > threshold)]
        r = {}
        if dets.shape[0] > 0:
            r['class'], r['dets'] = cls, dets
        else:
            r['class'], r['dets'] = cls, None
        res.append(r)
    return res   
    
def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def _draw_boxes_to_image(im, res):
    colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
              (151, 0, 255), (243, 223, 48), (0, 117, 255),\
              (58, 184, 14), (86, 67, 140), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (86, 255, 234)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.copy(im)
    cnt = 0
    for ind, r in enumerate(res):
        if r['dets'] is None: continue
        dets = r['dets']
        for i in range(0, dets.shape[0]):
            (x1, y1, x2, y2, score) = dets[i, :]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[ind % len(colors)], 2)
            text = '{:s} {:.2f}'.format(r['class'], score)
            cv2.putText(image, text, (x1, y1), font, 0.6, colors[ind % len(colors)], 1)
            cnt = (cnt + 1)
    return image

def _draw_gt_to_image(im, gt_boxes):
    image = np.copy(im)

    for i in range(0, gt_boxes.shape[0]):
        (x1, y1, x2, y2, score) = gt_boxes[i, :]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return image

def _draw_dontcare_to_image(im, dontcare):
    image = np.copy(im)

    for i in range(0, dontcare.shape[0]):
        (x1, y1, x2, y2) = dontcare[i, :]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return image

def train_net(network, imdb, roidb, output_dir, log_dir,
              pretrained_model=None, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, log_dir, pretrained_model)
        print 'Solving...'
        sw.train_model(sess, cfg.TRAIN.MAX_ITER, restore=restore)
        print 'done solving'