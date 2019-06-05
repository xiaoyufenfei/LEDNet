#### Training LEDNet in Pytorch

PyTorch code for training LEDNet model on Cityscapes. The code was based initially on the code from [bodokaiser/piwise](https://github.com/bodokaiser/piwise), adapted with several custom added modifications and tweaks. Some of them are:
- Load cityscapes dataset
- LEDNet model definition
- Calculate IoU on each epoch during training
- Save snapshots and best model during training
- Save additional output files useful for checking results (see below "Output files...")
- Resume training from checkpoint (use "--resume" flag in the command)

#### Options
For all options and defaults please see the bottom of the "main.py" file. Required ones are --savedir (name for creating a new folder with all the outputs of the training) and --datadir (path to cityscapes directory).

#### Example commands
Train encoder with 300+ epochs and batch=5 and then train decoder (decoder training starts after encoder training): for example
```
python main.py --savedir logs --datadir /home/datasets/cityscapes/ --num-epochs 300 --batch-size 5 ...
```

Each training will create a new folder in the "LEDNet_master/save/" directory named with the parameter --savedir and the following files:
* **{model}.py**: copy of the model file used (default lednet.py). 
* **model.txt**: Plain text that displays the model's layers
* **model_best.pth**: saved weights of the epoch that achieved best val accuracy.
* **model_best.pth.tar**: Same parameters as "checkpoint.pth.tar" but for the epoch with best val accuracy.
* **opts.txt**: Plain text file containing the options used for this training
* **automated_log.txt**: Plain text file that contains in columns the following info of each epoch {Epoch, Train-loss,Test-loss,Train-IoU,Test-IoU, learningRate}. Can be used to plot using Gnuplot or Excel or Matplotlib.
* **best.txt**: Plain text file containing a line with the best IoU achieved during training and its epoch.
* **checkpoint.pth.tar**: bundle file that contains the checkpoint of the last trained epoch, contains the following elements: 'epoch' (epoch number as int), 'arch' (net definition as a string), 'state_dict' (saved weights dictionary loadable by pytorch), 'best_acc' (best achieved accuracy as float), 'optimizer' (saved optimizer parameters).

NOTE: Encoder trainings have an added "_encoder" tag to each file's name.

#### IoU display during training

NEW: In previous code, IoU was calculated using a port of the cityscapes scripts, but new code has been added in "iouEval.py" to make it class-general, non-dependable on other code, and much faster (using cuda)

By default, only Validation IoU is calculated for faster training (can be changed in options)

#### Visualization
If you want to visualize the outputs during training add the "--visualize" flag and open an extra tab with:
```
python -m visdom.server -port 8097
```
The plots will be available using the browser in http://localhost.com:8097

#### Multi-GPU
If you wish to specify which GPUs to use, use the CUDA_VISIBLE_DEVICES command:
```
CUDA_VISIBLE_DEVICES=0 python main.py ...
CUDA_VISIBLE_DEVICES=0,1 python main.py ...
```


