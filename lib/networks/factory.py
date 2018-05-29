from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train
from .Retinanet_test import Retinanet_test
from .Retinanet_train import Retinanet_train
__sets = {}

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'Retinanet':
        if name.split('_')[1] == 'test':
           return Retinanet_test()
        elif name.split('_')[1] == 'train':
           return Retinanet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
