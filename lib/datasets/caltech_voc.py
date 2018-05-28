#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:34:48 2017

@author: heichang
"""

from .imdb import imdb
import os
from ..fast_rcnn.config import cfg
import cPickle
import numpy as np
import xml.etree.ElementTree as ET

class caltech_voc(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'caltech_'+image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__', 'pedestrian')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)        
        
        
        
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'CALVOC') 
       
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        care_inds = np.empty((0), dtype=np.int32)
        dontcare_inds = np.empty((0), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            class_name = obj.find('name').text.lower().strip()
            if class_name != 'dontcare':
                care_inds = np.append(care_inds, np.asarray([ix], dtype=np.int32))
            if class_name == 'dontcare':
                dontcare_inds = np.append(dontcare_inds, np.asarray([ix], dtype=np.int32))
                boxes[ix, :] = [x1, y1, x2, y2]
                continue
            cls = self._class_to_ind[class_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        # deal with dontcare areas
        dontcare_areas = boxes[dontcare_inds, :]
        boxes = boxes[care_inds, :]
        gt_classes = gt_classes[care_inds]

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'dontcare_areas' : dontcare_areas,
                'flipped' : False}