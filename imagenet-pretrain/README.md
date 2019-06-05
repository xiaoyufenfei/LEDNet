#### Model and ImageNet pretraining script 

This folder contains the script and model definition to pretrain LEDNet's encoder in ImageNet Data. 

The script is an adaptation from the code in [Pytorch Imagenet example](https://github.com/pytorch/examples/tree/master/imagenet). Please make sure that you have ImageNet dataset split in train and val folders before launching the script. Refer to that repository for instructions about usage and main.py options. Basic command:

```
python main.py <imagenet_folder_path>
```


#### Third Party Project Reference

- [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet)

- [ShuffleNetv2 in PyTorch](https://github.com/Randl/ShuffleNetV2-pytorch)

