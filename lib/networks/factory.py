from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train
from .Retinanet_test import Retinanet_test
from .Retinanet_train import Retinanet_train
from .Atrousnet_test import Atrousnet_test
from .Atrousnet_train import Atrousnet_train
from .MSnet_test import MSnet_test
from .MSnet_train import MSnet_train
__sets = {}

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown network: {}'.format(name))
    elif name.split('_')[0] == 'Retinanet':
        if name.split('_')[1] == 'test':
           return Retinanet_test()
        elif name.split('_')[1] == 'train':
           return Retinanet_train()
        else:
           raise KeyError('Unknown network: {}'.format(name))
    elif name.split('_')[0] == 'Atrousnet':
        if name.split('_')[1] == 'test':
           return Atrousnet_test()
        elif name.split('_')[1] == 'train':
           return Atrousnet_train()
        else:
           raise KeyError('Unknown network: {}'.format(name))
    elif name.split('_')[0] == 'MSnet':
        if name.split('_')[1] == 'test':
           return MSnet_test()
        elif name.split('_')[1] == 'train':
           return MSnet_train()
        else:
           raise KeyError('Unknown network: {}'.format(name))
    else:
        raise KeyError('Unknown network: {}'.format(name))
