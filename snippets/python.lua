local ls = require("luasnip") --{{{
local s = ls.s --> snippet
local i = ls.i --> insert node
local t = ls.t --> text node

local d = ls.dynamic_node
local c = ls.choice_node
local f = ls.function_node
local sn = ls.snippet_node

local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep

local snippets, autosnippets = {}, {} --}}}

local group = vim.api.nvim_create_augroup("type de fichier python ", { clear = true })
local file_pattern = "*.py"

local function cs(trigger, nodes, opts) --{{{
    local snippet = s(trigger, nodes)
    local target_table = snippets

    local pattern = file_pattern
    local keymaps = {}

    if opts ~= nil then
        -- check for custom pattern
        if opts.pattern then
            pattern = opts.pattern
        end

        -- if opts is a string
        if type(opts) == "string" then
            if opts == "auto" then
                target_table = autosnippets
            else
                table.insert(keymaps, { "i", opts })
            end
        end

        -- if opts is a table
        if opts ~= nil and type(opts) == "table" then
            for _, keymap in ipairs(opts) do
                if type(keymap) == "string" then
                    table.insert(keymaps, { "i", keymap })
                else
                    table.insert(keymaps, keymap)
                end
            end
        end

        -- set autocmd for each keymap
        if opts ~= "auto" then
            for _, keymap in ipairs(keymaps) do
                vim.api.nvim_create_autocmd("BufEnter", {
                    pattern = pattern,
                    group = group,
                    callback = function()
                        vim.keymap.set(keymap[1], keymap[2], function()
                            ls.snip_expand(snippet)
                        end, { noremap = true, silent = true, buffer = true })
                    end,
                })
            end
        end
    end

    table.insert(target_table, snippet) -- insert snippet into appropriate table
end --}}}


-- Ecrire ses snippets lua on peut utiliser le snipet luasnippet 
cs("pythontestsnippet", fmt( -- python test snippet
[[Python test snippet;]], {}))

cs("matplot", fmt( -- import matplotlib to plot stuff
[[
import matplotlib.pyplot as plt 
]], {}))



-- snippets pour PYTORCH
cs("torch_pack", fmt( -- package pour PYTORCH 
[[
# recuperation des packages
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# construire les data pour la classification:
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
]], {
  }))


cs("cuda_check", fmt( -- check cuda with torch 
[[
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
]], {
  }))



cs("torch_dataset_simp", fmt( -- simple dataset pytorch
[[
class CustomDataset(Dataset):
    def __init__(self, img_folder, label_folder, classes, transforms):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.classes = classes
        self.img_paths = os.listdir(img_folder)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # charger l'image
        # charger les annotations
        # Convert the annotations to PyTorch tensors
        return img, targets
]], {
  }))


cs("torch_dataloader", fmt( -- build data loader from custom dataset
[[
data_loader = torch.utils.data.DataLoader(
dataset, batch_size=2, shuffle=True, num_workers=4,
collate_fn=utils.collate_fn)
]], {
  }))



cs("torch_transfo_simp", fmt( -- torch transfo simp exemple
[[
# transfo pour resize les images pour le reseau de neurone
my_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
]], {
  }))


cs("torch_dataset_classification_images", fmt( -- torch dataset classfication object
[[
# creation du dataset pour les images
dataset = datasets.ImageFolder(root='dataset', transform=my_transform)
]], {
  }))


cs("torch_split_dataset_simp", fmt( -- torch exemple pour split le dataset 
[[
# creation des datasets pour entrainement et validation
from sklearn.model_selection import train_test_split
# on genere trois dataset train_set, val_set et test_set
train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=42)
]], {
  }))


cs("torch_dataloader_for_each_dataset_simp", fmt( -- torch example dataloaders for training
[[
# generation des loaders pour faire entrainement en batch pour utiliser le gpu
import torch.utils.data as data
train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=32, shuffle=True)
]], {
  }))




cs("torch_display_images_from_loader", fmt( -- torch exemple to show images from loader 
[[
# Afficher plusieurs images avec leurs étiquettes correspondantes
for images, labels in train_loader:
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    for i, ax in enumerate(axs):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(dataset.classes[labels[i] ])
    plt.show()
    break
]], {
  }))




cs("torch_classification_finetuning_model_simp", fmt( -- torch classification model
[[
import torch
import torchvision.models as models
# model pretrained
model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
# remplacer la derniere couche
num_classes = 3
# definir les proprietes des la derniere couche
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
# geler les poids du network
for param in model.parameters():
    param.requires_grad = False
# modifier la derniere couche
for param in model.fc.parameters():
    param.requires_grad = True
]], {
  }))



cs("torch_classfication_training", fmt( -- torch exemple training simple for classification 
[[
# definition de la fonction de perte 
criterion = torch.nn.CrossEntropyLoss()
# definition de optimiseur avec learning rate faible car pretrained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# nombre epochs tour complet du dataset
num_epochs = 10
# boucle entrainement
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("epoch : ",epoch)
# save model for deployment
torch.save(model,"modeltrained.pt")
]], {
  }))



cs("torch_classification_training_detailed", fmt( -- torch exemple pour la classification avec print detaille
[[
import matplotlib.pyplot as plt
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    # Validation
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
    # Affichage de la perte moyenne pour chaque ensemble
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {{epoch}}: train_loss={{avg_train_loss:.4f}}, val_loss={{avg_val_loss:.4f}}")
# Traçage des pertes sur les ensembles d'entraînement et de validation
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Evolution of training and validation loss')
plt.show()
]], {}))


cs("torch_classification_deploy_simp", fmt( -- exemple de deployment classification 
[[
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
# load model
model = torch.load("./modeltrained.pt")
# eval mode
model.eval()
# test image
test_image = Image.open("./dataset/pipe/pipe_imgL100.jpg")
# transfo
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])
# apply transfo to test image
test_image = transform(test_image)
# Ajouter une dimension pour le lot pour appliquer le model
test_image = test_image.unsqueeze(0)
# faire une prediction
with torch.no_grad():
    outputs = model(test_image)
    _, predicted = torch.max(outputs.data, 1)
# Afficher l'image de test et sa prédiction
print("La prédiction pour cette image est:", predicted.item())
]], {
  }))



cs("dataset_exemple_classificaiton", fmt( -- exemple de format csv pour le dataset pour la classification
[[
barrage/barrage_imgL248.jpg,barrage
barrage/barrage_imgL418.jpg,barrage
]], {
  }))



cs("torch_custom_detection_dataset", fmt( -- exemple simple de dataset custom pour la detection avec pytorch 
[[
class DetectionDataset(Dataset):
    def __init__(self, img_folder, label_folder, classes, transforms):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.classes = classes
        self.img_paths = os.listdir(img_folder)
        self.transforms = transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # charger l'image
        img_path = os.path.join(self.img_folder, self.img_paths[idx])
        img = Image.open(img_path).convert('RGB')
        # charger les annotations
        label_path = os.path.join(self.label_folder, os.path.splitext(self.img_paths[idx])[0] + '.txt')
        with open(label_path, 'r') as f:
            labels = f.readlines()[1:]
        # Convert the annotations to PyTorch tensors
        bboxes = []
        labels_idx = []
        for label in labels:
            label_parts = label.strip().split()
            label = label_parts[-1]
            bbox = [int(x) for x in label_parts[:-1] ]
            bboxes.append(bbox)
            #print("label : ",label)
            labels_idx.append(class_to_num[label])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels_idx = torch.tensor(labels_idx, dtype=torch.long)       
        # adapt bboxes
        new_width = 300
        new_height = 300
        old_width, old_height = img.size
        r_width = new_width/old_width
        r_height = new_height/old_height
        # prétraiter l'image
        transform = transforms.Compose([
            transforms.Resize((new_width, new_height)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        bboxes[:,0] *= r_width
        bboxes[:,2] *= r_width
        bboxes[:,1] *= r_height
        bboxes[:,3] *= r_height
        targets = {{}}
        targets['boxes'] = bboxes
        targets['labels'] = labels_idx
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        return img, targets
]], {
  }))



cs("torch_detection_definition_dataset", fmt( -- torch definition dataset for detection
[[
classes = ['chain', 'fish', 'lace']
img_folder = './dataset/images'
label_folder = './dataset/labels'
dataset = DetectionDataset(img_folder, label_folder, classes, transforms=None)
]], {
  }))


cs("torch_detection_definition_dataloader", fmt( -- torch definition dataloader for detection 
[[
import utils
batch_size = 2
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
]], {
  }))


cs("torch_detection_definition_model", fmt( -- torch definition model for detection 
[[
# definition du model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_instance_segmentation_model(num_classes):
    # loading an instance segmentation model pre-trained on the  COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replacing the pre-trained head with the new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now we calculate the number of input features for the mask classifier
    return model
model = get_instance_segmentation_model(3)
]], {
  }))


cs("torch_detection_try_model_with_loader", fmt( -- exemple detection model with loader 
[[
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{{k: v for k, v in t.items()}} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
]], {
  }))


cs("torch_detection_definition_transfo_augmentation_for_training", fmt( -- exemple detection definition transfo for training 
[[
import utils
import transforms as T
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    #transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
]], {
  }))


cs("torch_detection_split_dataset", fmt( -- torch detection split dataset 
[[
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = DetectionDataset(img_folder, label_folder, classes, get_transform(train=True))
dataset_test = DetectionDataset(img_folder, label_folder, classes, get_transform(train=False))
import torch.utils.data
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
]], {
  }))


cs("torch_detection_split_dataloader", fmt( -- torch detection split dataloader 
[[
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
]], {
  }))


cs("torch_device_definition", fmt( -- torch define device to use 
[[
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
]], {
  }))


cs("torch_detection_definemodelandmovetodevice", fmt( -- torch model move to device 
[[
# get the model using our helper function
num_classes = 3
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
]], {
  }))


cs("torch_detection_lrscheduler", fmt( -- torch detection lr scheduler 
[[
# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
]], {
  }))


cs("torch_detection_training", fmt( -- torch detection training 
[[
# let's train it for 10 epochs
from torch.optim.lr_scheduler import StepLR
num_epochs = 10
import torch.optim as optim
import time
# define one epoch of training
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{{k: v.to(device) for k, v in t.items()}} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if batch_idx % print_freq == 0:
            print(f"Epoch [{{epoch}}] Batch [{{batch_idx}}/{{len(data_loader)}}] Loss: {{losses}}")
# run on all epochs
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
# save model
torch.save(model,"modeltrainend.pt")
]], {
  }))


cs("torch_detection_deploy", fmt( -- torch detection deploy 
[[
# load packages
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# load model
model = torch.load("./modeltrainend.pt")
# put model in eval mode
model.eval()
# Charger l'image de test
img = Image.open("./dataset/test/imgL1048.jpg")
# Appliquer les mêmes transformations que lors de l'entraînement
transform = transforms.Compose([
    transforms.ToTensor(),
])
# transformer img pour utiliser sur le model
test_image = transform(img)
# inference
output = model(test_image.unsqueeze(0).to('cuda'))
# recuperation de la bbox la plus probable
bbox = output[0]['boxes'][0].to('cpu').detach()
# create corner bbox
def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3)
# show corner bbox
def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
# dislay prediction
show_corner_bb(img, bbox)
plt.show()
]], {
  }))


cs("torch_segmentation_classnum", fmt( -- torch segmentation class to num def
[[
class_to_num = {{
        "chain" : 1,
        "fish" : 2,
        "background" : 3
        }}
for a,b in enumerate(class_to_num):
    print(a," : ",b)
print(class_to_num["chain"])
]], {
  }))


cs("torch_segmentation_custom_dataset", fmt( -- torch segmentation custom dataset 
[[
class SegmentationDataset(Dataset):
    def __init__(self, img_folder, label_folder, classes, transforms):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.classes = classes
        self.img_paths = os.listdir(img_folder)
        self.transforms = transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # charger l'image
        img_path = os.path.join(self.img_folder, self.img_paths[idx])
        img = Image.open(img_path).convert('RGB')
        mask_path = img_path.replace('images','masksclean').replace('jpg','png')
        mask = Image.open(mask_path)
        # charger les annotations
        label_path = os.path.join(self.label_folder, os.path.splitext(self.img_paths[idx])[0] + '.txt')
        with open(label_path, 'r') as f:
            labels = f.readlines()[1:]
        # Convert the annotations to PyTorch tensors
        bboxes = []
        labels_idx = []
        for label in labels:
            label_parts = label.strip().split()
            label = label_parts[-1]
            bbox = [int(x) for x in label_parts[:-1] ]
            bboxes.append(bbox)
            labels_idx.append(class_to_num[label])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels_idx = torch.tensor(labels_idx, dtype=torch.long)       
        # adapt bboxes
        new_width = 300
        new_height = 300
        old_width, old_height = img.size
        r_width = new_width/old_width
        r_height = new_height/old_height
        # prétraiter l'image
        transform = transforms.Compose([
            transforms.Resize((new_width, new_height)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        # conversion du mask
        transfo = transforms.ToTensor()
        masktorch = transfo(mask)
        maskbin = masktorch.sum(dim=0)
#        plt.imshow(maskbin)
#        plt.show()
#        print(maskbin.unique())
        maskstmp = torch.zeros((len(maskbin.unique()), maskbin.shape[0], maskbin.shape[1]))
        for id, val in enumerate(maskbin.unique()):
            maskstmp[id][maskbin == val] = id
#        print(maskstmp.shape)
        transforesize = transforms.Resize((new_width, new_height))
        masks = transforesize(maskstmp)
#        print(masks.shape)
        masksf = torch.zeros((masks.shape[0]-1,masks.shape[1],masks.shape[2]))
        for id in range(masks.shape[0]-1):
            masksf[id] = masks[id+1]
#        print(masksf.shape)
        bboxes[:,0] *= r_width
        bboxes[:,2] *= r_width
        bboxes[:,1] *= r_height
        bboxes[:,3] *= r_height
        targets = {{}}
        targets['boxes'] = bboxes
        targets['labels'] = labels_idx.flipud()
        targets['masks'] = masksf
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        return img, targets
]], {
  }))



cs("torch_segmentation_define_dataset_dataloader", fmt( -- torch segmenation define dataset and dataloader 
[[
classes = ['chain', 'fish', 'lace']
img_folder = './dataset/images'
label_folder = './dataset/labels'
dataset = SegmentationDataset(img_folder, label_folder, classes, transforms=None)
import utils
batch_size = 2
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

]], {
  }))



cs("torch_segmentation_model", fmt( -- torch segmentation model 
[[
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
def get_instance_segmentation_model(num_classes):
    # loading an instance segmentation model pre-trained on the  COCO dataset
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replacing the pre-trained head with the new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now we calculate the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # we also replace the mask predictor with a new mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
# define model with 3 classes
model = get_instance_segmentation_model(3)
model.to(device)
]], {
  }))


cs("torch_segmentation_try_model", fmt( -- torch segmentation try model 
[[
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{{k: v for k, v in t.items()}} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
]], {
  }))


cs("torch_segmentation_transfo_augmentation_train_dataset", fmt( -- torch segmentation transfo def and train dset def
[[
import utils
import transforms as T
# transfo only if training
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = SegmentationDataset(img_folder, label_folder, classes, get_transform(train=True))
#dataset_test = SegmentationDataset(img_folder, label_folder, classes, get_transform(train=False))
import torch.utils.data
#dataset = torch.utils.data.Subset(dataset, indices[:-50])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
]], {
  }))



cs("torch_segmentation_define_modeltodevice", fmt( -- torch segmentation define model to device
[[
# get the model using our helper function
num_classes = 3
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)
]], {
  }))


cs("torch_segmentation_lrscheduler", fmt( -- torch segmentation lr scheduler 
[[
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
]], {
  }))



cs("torch_segmentation_training", fmt( -- torch segmentation training 
[[
# let's train it for 10 epochs
from torch.optim.lr_scheduler import StepLR
num_epochs = 10
# import package for optimizer
import torch.optim as optim
import time
# define one epoch for training
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{{k: v.to(device) for k, v in t.items()}} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if batch_idx % print_freq == 0:
            print(f"Epoch [{{epoch}}] Batch [{{batch_idx}}/{{len(data_loader)}}] Loss: {{losses}}")
# define training loop
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
# save model
torch.save(model,"modeltrainend.pt")
]], {}))



cs("torch_segmentation_deploy", fmt( -- torch segmentation deploy 
[[
# import packages
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# import model
model = torch.load("./modeltrainend.pt")
# put model to eval mode
model.eval()
# Charger l'image de test
img = Image.open("./dataset/test/imgL1048.jpg")
# Appliquer les mêmes transformations que lors de l'entraînement
transform = transforms.Compose([
    transforms.ToTensor(),
])
# transfo image for inference
test_image = transform(img)
# inference
output = model(test_image.unsqueeze(0).to('cuda'))
# display prediction
fig, axes = plt.subplots(1,2)
axes[0].imshow(img)
axes[1].imshow(output[0]['masks'][0][0].to('cpu').detach())
plt.show()
# display bbox also predicted by the model
bbox = output[0]['boxes'][0].to('cpu').detach()
# create corner bbox
def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3)
# show corner bbox
def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
# display bbox
show_corner_bb(img, bbox)
plt.show()
]], {
  }))



cs("qt_simple_view", fmt( -- exemple pyqt dune view 
[[
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 300)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, 50, 91, 16))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 100, 89, 25))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Hello World!"))
        self.pushButton.setText(_translate("MainWindow", "Click me!"))
]], {
  }))


cs("qt_simple_controller", fmt( -- exemple pyqt controller simple
[[
from PyQt5 import QtWidgets
from ui_mainwindow import Ui_MainWindow
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self._init_ui()
    def _init_ui(self):
        self.pushButton.clicked.connect(self._on_button_clicked)
    def _on_button_clicked(self):
        self.label.setText("Button clicked!")
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 
]], {
  }))


cs("qt_mainwindowview_def", fmt( -- definition mainwindow qt 
[[
# window size
MainWindow.resize(719, 465)
# central widget
self.centralwidget = QtWidgets.QWidget(MainWindow)
]], {
  }))


cs("qt_vertical_layout", fmt( -- definition vertical layout qt 
[[
self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
]], {
  }))


cs("qt_horizontal_layout", fmt( -- definition horizontal layout qt 
[[
self.horizontalLayout = QtWidgets.QHBoxLayout()
]], {
  }))


cs("qt_label_definition", fmt( -- definition label 
[[
self.label = QtWidgets.QLabel(self.centralwidget)
font = QtGui.QFont()
font.setPointSize(13)
self.label.setFont(font)
# ajout du label au layout souhaite
#self.layout.addWidget(self.label)
]], {
  }))


cs("qt_edit_definition", fmt( -- definition edit texte 
[[
self.edit = QtWidgets.QLineEdit(self.centralwidget)
font = QtGui.QFont()
font.setPointSize(13)
self.edit.setFont(font)
# ajout du widget edit au layout
#self.layout.addWidget(self.edit)
]], {
  }))


cs("qt_toolbox_definition", fmt( -- definition toolbox definition 
[[
self.toolbutton = QtWidgets.QToolButton(self.centralwidget)
font = QtGui.QFont()
font.setPointSize(13)
self.toolbutton.setFont(font)
# ajout du widget au layout 
#self.layout_tocrypt.addWidget(self.toolbutton)
]], {
  }))


cs("qt_pushbutton_definition", fmt( -- definition pushbutton 
[[
self.button = QtWidgets.QPushButton(self.centralwidget)
font = QtGui.QFont()
font.setPointSize(13)
font.setBold(True)
self.button.setFont(font)
# ajout du bouton au layout vertical global
#self.verticalLayout.addWidget(self.button)
]], {
  }))


cs("qt_groupbox_definition", fmt( -- definition groupbox 
[[
self.groupbox = QtWidgets.QGroupBox(self.centralwidget)
font = QtGui.QFont()
font.setPointSize(13)
self.groupbox.setFont(font)
# ajout de layout et bouton au group
self.verticalLayout = QtWidgets.QVBoxLayout(self.groupbox)
# bouton radio
#self.button = QtWidgets.QRadioButton(self.groupbox)
# group de boutons radio
#self.groupbutton = QtWidgets.QButtonGroup(MainWindow)
# ajout du bouton radio au group
#self.groupbutton.addButton(self.button)
# ajout du bouton radio au layout vertical
#self.verticalLayout.addWidget(self.button)
# ajout du groupbox au layout vertical global
#self.verticalLayoutGlobal.addWidget(self.groupbox)
]], {
  }))


cs("qt_groupbutton_definition", fmt( -- definition button group 
[[
self.groupbutton = QtWidgets.QButtonGroup(MainWindow)
# ajout du bouton radio au group
#self.groupbutton.addButton(self.button)
]], {
  }))


cs("qt_translate_view", fmt( -- qt translate buttons labels 
[[
self.retranslateUi(MainWindow)
]], {
  }))


cs("qt_connect_button_to_controller", fmt( -- qt connect button 
[[
self.button.clicked.connect(MainWindow.button_clicked) # type: ignore
]], {
  }))


cs("qt_connect_radio_button", fmt( -- qt connect radio buttons
[[
QtCore.QMetaObject.connectSlotsByName(MainWindow)
]], {
  }))


cs("qt_translate_definition", fmt( -- qt define translate ui 
[[
def retranslateUi(self, MainWindow):
    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate("MainWindow", "IVM-Encrypter"))
    # exemple de naming
    #self.label.setText(_translate("MainWindow", "IVM-Encrypter"))
    #self.button.setText(_translate("MainWindow", "..."))
    #self.groupbox.setTitle(_translate("MainWindow", "Action à effectuer"))
]], {
  }))


cs("qt_view_packages", fmt( -- qt view packages import 
[[
from PyQt5 import QtCore, QtGui, QtWidgets
]], {
  }))


cs("qt_controller_packages", fmt( -- qt controller packages import 
[[
# package python
import os
import shutil
import subprocess
# package Qt
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
# link controller to view
#from mainwindow_view import Ui_MainWindow
]], {
  }))



cs("qt_controller_class_definition", fmt( -- qt controller class definition 
[[
# class fenetre herite de Ui_MainWindow la view
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
        # methode init 
        def __init__(self, parent=None):
            super(MainWindow, self).__init__(parent)
            self.setupUi(self)
            self._init_ui()
        # methode init_ui
        def _init_ui(self):
            self.button_crypt_only.setChecked(True)
            self.label_result.hide()
            self.button_crypt.setEnabled(False)
        # bouton exemple nom de la methode a def dans la view en connectant le bouton
#        def button_clicked(self):
#            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir...", '/')
#            # adapter le champs texte
#            self.edit.setText(directory)
#            if len(self.edit.text()):
#                print("successfully adapted text label")

]], {
  }))



cs("qt_run_app", fmt( -- qt run app
[[
# import qt packages
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
# import controller
from mainwindow_controller import MainWindow
# utils
import sys
# app definition
app = QtWidgets.QApplication(sys.argv)
# main window
window = MainWindow()
window.show()
# Start the event loop.
app.exec()
]], {
  }))



cs("qt_run_app_2", fmt( -- qt run app 2
[[
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 
]], {
  }))



cs("qt_view_class_definition", fmt( -- qt view class definition 
[[
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 300)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

#        self.centralwidget = QtWidgets.QWidget(MainWindow)
#        self.label = QtWidgets.QLabel(self.centralwidget)
#        self.retranslateUi(MainWindow)
#        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.pushButton.setText(_translate("MainWindow", "Click me!"))
]], {
  }))



cs("qt_view_label_translate_text", fmt( -- qt define label text into view 
[[
self.label.setText(_translate("MainWindow", "Hello World!"))
]], {
  }))



cs("qt_alignement_exemple", fmt( -- qt exemple pour aligner les widgets 
[[
# on aligne le layout top et centrer horizontalement
self.verticalLayout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
]], {
  }))



cs("qt_controller_find_directory", fmt( -- qt controller find directory button
[[
directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir...", '/')
]], {
  }))


cs("qt_controller_find_file", fmt( -- qt controller find file button 
[[
file, _= QtWidgets.QFileDialog.getOpenFileName(None, "Select File")
]], {
  }))


cs("network_packages", fmt( -- packages pour network with python 
[[
import sys
import os
import time
import requests
import json
import struct
import socket
]], {
  }))


cs("network_connexion_to_server", fmt( -- connexion au server 
[[
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect(("192.168.51.113", 1234))
]], {
  }))


cs("network_loop_receive_images", fmt( -- loop pour recevoir les images 
[[
# Tant que la connexion est active
while socket:
    # Receive image size
    data = socket.recv(struct.calcsize('<L'))
    if not data:
        print("pas de donnees on quitte")
        #continue
        break
    size = struct.unpack('<L', data)[0]

    # Receive image data
    data = bytearray(size)
    view = memoryview(data)
    while size:
        n_bytes = socket.recv_into(view, size)
        view = view[n_bytes:]
        size -= n_bytes

    # Decode images left and right
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        h,w,ch = img.shape
        cv2.imshow('Client Window', img / 255.0)
        cv2.waitKey(1)
    except cv2.error as e:
        print("Failed to decode image:", e)
        break
        #continue
]], {
  }))



cs("network_client_with_qt_and_thread_example", fmt( -- network client python mixing qt and thread
[[
import sys
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import struct
import socket

class ImageReceiver(QObject):
    image_received = pyqtSignal(object) # signal pour emettre une image

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = None
        self.fps = 0
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.start_time = cv2.getTickCount()
        self.socket.connect((self.host, self.port))
    def receive(self):
        # Tant que la connexion est active
        while self.socket:
            # Receive image size
            data = self.socket.recv(struct.calcsize('<L'))
            if not data:
                break
            size = struct.unpack('<L', data)[0]
            # Receive image data
            data = bytearray(size)
            view = memoryview(data)
            while size:
                n_bytes = self.socket.recv_into(view, size)
                view = view[n_bytes:]
                size -= n_bytes
            # Decode image
            try:
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

                self.fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - self.start_time))
                self.start_time = cv2.getTickCount()
            except cv2.error as e:
                print("Failed to decode image:", e)
                continue

            # Emit signal with image
            self.image_received.emit((img, self.fps))
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the label to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        # Set up the label to display the FPS
        self.fps_label = QLabel(self)
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.fps_label.setStyleSheet("background-color: rgba(255, 255, 255, 50); padding: 5px;")

#        self.label = QLabel(self)
#        self.setCentralWidget(self.label)
        # image recceiver
        self.image_receiver = ImageReceiver('localhost', 1234)
        self.image_receiver.image_received.connect(self.show_image)
        self.image_receiver_thread = None
    def start_receiver(self):
        if not self.image_receiver_thread:
            self.image_receiver.connect()
            self.image_receiver_thread = QThread()
            self.image_receiver.moveToThread(self.image_receiver_thread)
            self.image_receiver_thread.started.connect(self.image_receiver.receive)
            self.image_receiver_thread.start()
    def stop_receiver(self):
        if self.image_receiver_thread:
            self.image_receiver.disconnect()
            self.image_receiver_thread.quit()
            self.image_receiver_thread.wait()
            self.image_receiver_thread = None
    def show_image(self, data):
        img, fps = data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()
        # Update the FPS label text
        self.fps_label.setText(f"FPS: {{fps}}")
        # Resize the FPS label to fit its contents
        self.fps_label.adjustSize()
        # Position the FPS label in the top-right corner of the image label
        self.fps_label.move(self.image_label.width() - self.fps_label.width() - 10, 10)
    def closeEvent(self, event):
        self.stop_receiver()
        super().closeEvent(event)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    main_window.start_receiver()
    sys.exit(app.exec_())
]], {
  }))







-- Tutorial Snippets go here --

--#region
--
-- End Refactoring --

return snippets, autosnippets

