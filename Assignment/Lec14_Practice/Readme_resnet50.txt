Readme_resnet50_skeleton.py
2071044 정민정
* Purpose of this code
: This code implements a model "ResNet50". This code uses CIFAR10 dataset.
* Class and Functions
1. ResidualBlock class
- represent an object of the Residual Block.
- apply 1x1 conv layer → 3x3 conv layer → 1x1 conv layer to the input image.
- If downsample=True, apply downsize method to input image.
- If downsample =False, apply make_equal_channel method to input image
  (only if the number of input image channels != the number of output image channels)
- Then return the value of output image + input image.
2. ResNet50_layer4 class
- represent an object of a ResNet50 model with four layers.
- apply layer1→layer2→layer3→layer4→AvgPool layer→FC layer to input image.
- layer 1: 7x7 conv layer(channel=64, stride=2) → 3x3 max pool layer(stride=2)
- layer 2: apply [1x1 conv layer(channel=64)→3x3 conv layer(channel=64)→1x1 conv layer(channel=256)] 2 times,
	and [1x1 conv layer(channel=64,stride=2)→3x3 conv layer(channel=64)→1x1 conv layer(channel=256)] 1 time.
- layer 3: apply [1x1 conv layer(channel=128)→3x3 conv layer(channel=128)→1x1 conv layer(channel=512)] 3 times,
	and [1x1 conv layer(channel=128,stride=2)→3x3 conv layer(channel=128)→1x1 conv layer(channel=512)] 1 time.
- layer 4: apply [1x1 conv layer(channel=256)→3x3 conv layer(channel=256)→1x1 conv layer(channel=1024)] 6 times.