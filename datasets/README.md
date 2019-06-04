#### Cityscapes dataset overview：

1. The data is **registered and downloaded **on the official website, https://www.cityscapes-dataset.com/. The official website can also view the indicators reached by everyone's neural network in the benchmarks.

2. data preprocessing and evaluation results code **download**: https://github.com/mcordts/cityscapesScripts

The original image is stored in the **leftImg8bit** folder, and the finely labeled data is stored in the **gtFine** (gt: ground truth) folder. The training set consists of **2975** trains and the validation set is **500** sheets (val), all of which have corresponding labels. But the test set (test) only gave the original picture, no label, the official used to evaluate the code submitted by everyone (to prevent someone from using the test set training brush indicator). Therefore, in actual use, you can use the validation set to do the test. Coarse labeled data is stored in the **gtCoarse** (gt: ground truth) folder.

Each image in the tag file corresponds to 4 files, where _gtFine_polygons.json stores the classes and corresponding regions (the boundary of the region is represented by the position of the vertices of the polygon); the value of _gtFine_labelIds.png is 0-33, different values Representing different classes, the correspondence between values and classes is defined in the code in cityscapesscripts/helpers/labels.py; _gtFine_instaceIds.png is an example split; _gtFine_color.png is for everyone to visualize, and the correspondence between different colors and categories is also Description in the labels.py file.

#### Dataset Structure:

```
├── datasets  # contains all datasets for the project
|  └── cityscapes #  cityscapes dataset
|  |  └── gtCoarse #  Coarse cityscapes annotation
|  |  └── gtFine #  Fine cityscapes annotation
|  |  └── leftImg8bit #  cityscapes training image
|  |  └── results #results move here for eval by evalPixelLevelSemanticLabeling.py 
|  └── cityscapesscripts #  cityscapes dataset label convert scripts！
|  |  └── annotation #  
|  |  └── evalution #  
|  |  └── helps #  
|  |  └── preparation #  
|  |  └── viewer #  
|  |  └── __init__.py #  

```

