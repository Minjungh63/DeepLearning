Readme_main.py
2071044 정민정
* Purpose of this code
: This code trains a model using pytorch program. This code uses CIFAR10 dataset.
* How to run this code
1. Move the file "resnet50_skeleton.py" and "vgg16_full.py" to the same directory.
2. Choose the model to train : VGG16 or ResNet50
3. If VGG16 is chosen, please add "from vgg16_full import *", "model = vgg16.to(device), and PATH=<vgg_epoch250.ckpt directory> to this code.
If ResNet50 is chosen, please add "from resnet50_skeleton import *", "model=ResNet50_layer4().to(device)", and PATH =<resnet50_epoch285.ckpt>
4. Choose the loss and optimize method.
5. Run the code.
