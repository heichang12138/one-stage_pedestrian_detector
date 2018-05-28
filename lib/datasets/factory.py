from .caltech_voc import caltech_voc

__sets = {}
for split in ['train','test','occ']:
    name = 'caltech_{}'.format(split)
    __sets[name] = (lambda split=split: caltech_voc(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        print (__sets.keys())
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()