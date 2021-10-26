lr = None
epochs = None
batch_size = None
base_directory = None

cyclic_min_lr = None
cyclic_max_lr = None
cyclic_epochs = None
adam_lr = None

autoaugment = None
augment = None
mean = None
cutout = None

train_N = None
test_N = None
trainset = None
testset = None
unsupset = None

def guard():
    for key, value in globals().items():
        if not callable(value) and not key.endswith('__'):
            assert value is not None, '%s=%r is None' % (key, value)