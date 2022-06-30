Readme_vgg16_full.py
2071044 정민정
* Purpose of this code
: This code implements a model "VGG16". This code uses CIFAR10 dataset.
* Class and Functions
1. VGG class
- represent an object of the VGG16 model.
- The layer information of VGG may be known through the cfg array.
- The sequence of layers is generated using the make_layer function.
2. make_layers() function
- generate layers of VGG16 model.
- If the value of cfg array element is 'M', create a max-pooling layer(kernel size=2, stride=2)
- If the value of cfg array element is not 'M', create a conv layer and ReLU layer.