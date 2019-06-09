#### Functions for evaluating/visualizing the network's output

Currently there are 4 usable functions to evaluate stuff:
- eval_cityscapes_color
- eval_cityscapes_server
- eval_iou
- eval_forward_time

#### eval_cityscapes_server.py 

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server. By default it saves images in eval/save_results/ folder.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**
```
python eval_cityscapes_server.py --datadir /xx/datasets/cityscapes/ --loadDir ../save/logs/ --loadWeights model_best.pth --loadModel lednet.py --subset val
```

#### eval_cityscapes_color.py 

This code can be used to produce segmentation of the Cityscapes images in color for visualization purposes. By default it saves images in eval/save_color/ folder. You can also visualize results in visdom with --visualize flag.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val', 'test', 'train' or 'demoSequence'). For other options check the bottom side of the file.

**Examples:**

```
python eval_cityscapes_color.py --datadir /xx/datasets/cityscapes/ --loadDir ../save/logs/ --loadWeights model_best.pth --loadModel lednet.py --subset val
```

#### eval_iou.py 

This code can be used to calculate the IoU (mean and per-class) in a subset of images with labels available, like Cityscapes val/train sets.

**Options:** Specify the Cityscapes folder path with '--datadir' option. Select the cityscapes subset with '--subset' ('val' or 'train'). For other options check the bottom side of the file.

**Examples:**

```
python eval_iou.py --datadir /xx/datasets/cityscapes/ --loadDir ../save/logs/ --loadWeights model_best.pth --loadModel lednet.py --subset val
```

#### eval_forward_time.py
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

**Options:** Option '--width' specifies the width (default: 1024). Option '--height' specifies the height (default: 512). For other options check the bottom side of the file.

**Examples:**
```
python eval_forward_time.py --batch-size=6
```



