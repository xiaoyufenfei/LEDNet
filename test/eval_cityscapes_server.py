import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes

from lednet import Net


from transform import Relabel, ToLabel, Colorize


NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512),
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform_cityscapes = Compose([
    Resize(512),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),  
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(255, 0),
    ToPILImage(),
    Resize(1024, Image.NEAREST),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


    model = Net(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            #labels = labels.cuda()

        inputs = Variable(images)
        #targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        #print (numpy.unique(label.numpy()))  #debug

        
        filenameSave = "./save_results/" + filename[0].split("leftImg8bit/")[1]
        
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        #image_transform(label.byte()).save(filenameSave)
        label_cityscapes.save(filenameSave)

        print (step, filenameSave)

    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../save/logs/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--subset', default="val")  #can be val, test, train, demoSequence
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
