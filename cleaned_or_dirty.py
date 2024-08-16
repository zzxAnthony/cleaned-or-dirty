import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, models
import os
import shutil
from tqdm import tqdm
import time
import copy

train_dir = 'train'
val_dir = 'train'
classsnames = ['cleaned' , 'dirty']

class Data_Module() :
    def __init__(self):
        self.get_data()
    def get_data(self) -> None:
        train_dir = 'train'
        val_dir = 'val'
        test_dir = 'test'
        classnames = ['cleaned' , 'dirty']
        for dirname in [train_dir, val_dir]:
            for class_name in classnames:
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        for class_name in classnames:
            src_dir = os.path.join("/kaggle/input/plates/plates", "train" , class_name)
            for i , filename in enumerate(os.listdir(src_dir)):
                if i % 5 != 0 :
                    dest_dir = os.path.join(train_dir, class_name)
                else :
                    dest_dir = os.path.join(train_dir, class_name)
                shutil.copy(os.path.join(src_dir , filename) , os.path.join(dest_dir , filename))
    def get_dataset(self , type_set='train' , transform=[] , batch_size=8) :
        if type_set in ['train', 'val']:
            return torchvision.datasets.ImageFolder(type_set, transforms)

    def get_dataloader(self, type_set='train', transforms=[], batch_size=8):
        if type_set == 'train':
            if transforms == []:
                dataset = torchvision.datasets.ImageFolder(type_set)
            elif len(transforms) == 1:
                dataset = torchvision.datasets.ImageFolder(type_set, transforms[0])
            else:
                dataset = torchvision.datasets.ImageFolder(type_set, transforms[0])
                for transform in transforms[1:]:
                    dataset += torchvision.datasets.ImageFolder(type_set, transform)
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
        elif type_set == 'val':
            if transforms == []:
                dataset = torchvision.datasets.ImageFolder(type_set)
            elif len(transforms) == 1:
                dataset = torchvision.datasets.ImageFolder(type_set, transforms[0])
            else:
                dataset = torchvision.datasets.ImageFolder(type_set, transforms[0])
                for transform in transforms[1:]:
                    dataset += torchvision.datasets.ImageFolder(type_set, transform)
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4)

data = Data_Module()
data.get_data()

train_transforms1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_transforms2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_transforms3 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_transforms4 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_transforms5 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms6 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomGrayscale(),
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=90, fill=255),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms7 = transforms.Compose([
    transforms.CenterCrop(180),
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms = [train_transforms2, train_transforms3, train_transforms1, train_transforms4, train_transforms5, train_transforms6, train_transforms7]
val_transforms = [val_transforms]

batch_size = 8

train_dataloader = data.get_dataloader(type_set='train',  transforms=train_transforms ,batch_size=8)
val_dataloader = data.get_dataloader(type_set='train',transforms=val_transforms , batch_size = batch_size)

X_batch , y_batch = next(iter(train_dataloader))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.255])

def train_model(model, loss , optimizer, scheduler , num_epochs):
    zzx = model
    ac = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ['train','val'] :
            if phase == 'train' :
                dataloader = train_dataloader
                scheduler.step()
                model.train()
            else :
                dataloader = train_dataloader
                model.eval()
            running_loss = 0
            running_corrects = 0
            for inouts , labels in tqdm(dataloader):
                inputs = inouts.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)
                    if phase == 'train' :
                        loss_value.backward()
                        optimizer.step()
                running_loss += loss_value.item()
                running_corrects += (preds_class == labels.data).float().mean()
            epoch_loss = running_loss / (len(dataloader))
            epoch_accuracy = running_corrects / (len(dataloader))
            if epoch_accuracy > ac :
                ac = epoch_accuracy
                zzx = model
            model = zzx
            print('{} Loss:{:.4f} Accuracy:{:.4f}'.format(phase, epoch_loss, epoch_accuracy))
    return zzx

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

for param in model.avgpool.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 2))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model , loss , optimizer , scheduler , num_epochs=20)

test_dir = 'test'
shutil.copytree(os.path.join('/kaggle/input/plates/plates', 'test'), os.path.join(test_dir, 'unknown'))


class TestImageDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(TestImageDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path
test_dataset = TestImageDataset('/kaggle/working/test', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=4)

def test_predict(model):
    model.eval()
    test_predictions = []
    test_img_paths = []
    for inputs, labels, paths in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
        test_img_paths.extend(paths)

    return test_img_paths, np.concatenate(test_predictions)
test_img_paths , test_predictions = test_predict(model)

def get_submission(thresh=0.5):
    df = pd.DataFrame.from_dict({'id':test_img_paths, 'label':test_predictions})
    df['label'] = df['label'].map(lambda x: 'dirty' if x >= thresh else 'cleaned')
    df['id'] = df['id'].str.replace('/kaggle/working/test/unknown/', '')
    df['id'] = df['id'].str.replace('.jpg' , '')
    df.set_index('id', inplace=True)
    df.to_csv('outputs.csv')

    return df

sb_df = get_submission(thresh=0.5)