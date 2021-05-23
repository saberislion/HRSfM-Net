HIGH_RESOLUTION_NET = [
    {'PRETRAINED_LAYERS': ['*'],
     'STEM_INPLANES':64,
     'FINAL_CONV_KERNEL':1,
     'WITH_HEAD':True},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 2,
     'NUM_BLOCKS': [4, 4],
     'NUM_CHANNELS': [64, 128],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM',
     'is_fuse':False},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 3,
     'NUM_BLOCKS': [4, 4, 4],
     'NUM_CHANNELS': [64, 128, 256],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM',
     'is_fuse':False},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 4,
     'NUM_BLOCKS': [4, 4, 4, 4],
     'NUM_CHANNELS': [64, 128, 256, 512],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM',
     'is_fuse':False},

    {'NUM_MODULES': 1,
         'NUM_BRANCHES': 5,
         'NUM_BLOCKS': [4, 4, 4, 4, 4],
         'NUM_CHANNELS': [64, 128, 256, 512, 1024],
         'BLOCK': 'BASIC',
         'FUSE_METHOD': 'SUM',
         'is_fuse':False},
]

