import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from mypath import Path
from networks.fcn import VGGNet, FCNs
from dataloaders.davis_2016 import DAVIS2016
import dataloaders.custom_transforms as tr

import os


use_cuda = 0
device = torch.device("cuda" if use_cuda else "cpu")

resume_epoch = 0
n_Epoch = 150
n_TestEpoch = 20
save_every = 10
print_every = 20

train_batch = 1

db_root_dir = Path.db_root_dir()
save_dir = Path.save_root_dir()

pretrain = True
modelName = 'parent'

if resume_epoch == 0:
    vgg_model = VGGNet(pretrained=True, required_grad=True)
    model = FCNs(pretrained_net=vgg_model, n_class=2)
else:
    vgg_model = VGGNet(pretrained=False, required_grad=True)
    model = FCNs(pretrained_net=vgg_model, n_class=2)
    model.load_state_dict(
        torch.load(os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))

model.to(device)

# if True:
#     for name, param in model.named_parameters():
#         print(name, param.size())

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = 5e-6)

composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
db_train = DAVIS2016(train=True, inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=train_batch, shuffle=True, num_workers=2)

db_test = DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor())
# testloader = DataLoader(db_test, batch_size=test_batch, shuffle=False, num_workers=2)
loss_list = []

for epoch in range(resume_epoch, n_Epoch):
    loss_running = 0
    for i, batch in enumerate(trainloader):
        inputs, gts = batch['image'], batch['gt']
        optimizer.zero_grad()
        output = model(inputs)
        output = torch.sigmoid(output)
        # print(output.shape, gts.shape)
        loss = criterion(output, gts)
        loss_running += loss.data.item()
        loss.backward()
        optimizer.step()
    loss_list.append(loss_running)
    if epoch % print_every == 0:
        print('Epoch: %d, loss %f' % (epoch, loss_list[-1]))
    if epoch % save_every == 0:
        print("Epoch %d, Saving model.." % epoch)
        torch.save(model.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))
