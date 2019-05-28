# One-stage Pedestrian Detector
Improve RPN for better pedestrian detection:
![](pipeline.png)

## Environment

- Ubuntu 16.04
- CUDA 8.0.61
- CUDNN 7005
- Tensorflow 1.4.0

## Dataset preparation
- Download the images from [caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
- Convert data to VOC format (refer to [TFFRCNN](https://github.com/CharlesShang/TFFRCNN/tree/master/experiments/scripts))
- Place the data (or create symlinks) to make the data folder like:
  ~~~
  ${ROOT}
  |-- data
  `-- |-- CALVOC
      `-- |-- Annotations
          |   |-- 000000.xml
          |   |-- 000001.xml
                  ...
          `-- JPEGImages
          |   |-- 000000.jpg
          |   |-- 000001.jpg
                  ...
          `-- ImageSets
              `-- |-- Main
              |   |-- test.txt
              |   |-- train.txt
  ~~~
 
## Get start
Run a demo:
~~~
python faster_rcnn/demo.py --net VGGnet_test --model output/rpn1457/VGGnet_iter_80000.ckpt --cfg cfgs/rpn_caltech_vgg.yml
~~~
Pretrained models can be download from [BaiduYun](https://pan.baidu.com/s/13gfH-7ZCpzzxH0iLjdVAGg)(password: 80j9), put them in `./output`.

If set `--cfg cfg/rpn_caltech_vgg.yml`, then `--net` must be `VGGnet_test/train`, otherwise use `--net MSnet_test/train`.

Evaluate on caltech:
~~~
python faster_rcnn/test_net.py --weights output/bfl1055/VGGnet_iter_80000.ckpt --cfg cfgs/rpn_caltech_ohem.yml --imdb caltech_test --network MSnet_test --vis
~~~

Train on caltech:
~~~
python faster_rcnn/train_net.py --weights data/pretrain_model/VGG_imagenet.npy --network MSnet_train --cfg cfgs/rpn_caltech_ms.yml
~~~

## Main results
On Caltech:
![](caltech.png)

On Caltech-new:
![](caltech_new.png)
