import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class Retinanet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        # classification subnet
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='clsSubnet_conv_1')
             .conv(3,3,512,1,1,name='clsSubnet_conv_2')
             .conv(3,3,512,1,1,name='clsSubnet_conv_3')
             .conv(3,3,512,1,1,name='clsSubnet_conv_4')
             .final_conv(3,3,len(anchor_scales)*2,1,1,padding='VALID',relu=False,name='rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_sigmoid(name='rpn_cls_prob')
             .spatial_reshape_layer(len(anchor_scales)*2, name = 'rpn_cls_prob_reshape'))
        # box regression subnet
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='boxSubnet_conv_1')
             .conv(3,3,512,1,1,name='boxSubnet_conv_2')
             .conv(3,3,512,1,1,name='boxSubnet_conv_3')
             .conv(3,3,512,1,1,name='boxSubnet_conv_4')
             .conv(3,3,len(anchor_scales)*4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))