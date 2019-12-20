import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from mypath import Path
from networks.fcn import VGGNet, FCNs
from dataloaders.davis_2016 import DAVIS2016
import dataloaders.custom_transforms as tr

import numpy as np
import matplotlib.pyplot as plt
import os


use_cuda = 0
device = torch.device("cuda" if use_cuda else "cpu")

parent_epoch = 1

n_Epoch = 0
print_every = 500

train_batch = 1

db_root_dir = Path.db_root_dir()
save_dir = Path.save_root_dir()

parentModelName = 'parent'


if 'SEQ_NAME' not in os.environ.keys():
    seq_name = 'blackswan'
else:
    seq_name = str(os.environ['SEQ_NAME'])

db_root_dir = Path.db_root_dir()
save_dir = Path.save_root_dir()

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))


vgg_model = VGGNet(pretrained=False, required_grad=True)
model = FCNs(pretrained_net=vgg_model, n_class=2)
model.load_state_dict(
    torch.load(os.path.join(save_dir, parentModelName + '_epoch-' + str(parent_epoch - 1) + '.pth'),
               map_location=lambda storage, loc: storage))

model.to(device)

lr = 1e-8
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)

composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])

db_train = DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
trainloader = DataLoader(db_train, batch_size=train_batch, shuffle=True, num_workers=1)

db_test = DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0

loss_list = []
print("Start of Online Training, sequence: " + seq_name)
for epoch in range(0, n_Epoch):
    loss_running = 0
    for i, batch in enumerate(trainloader):
        inputs, gts = batch['image'], batch['gt']
        optimizer.zero_grad()
        output = model(inputs)
        output = torch.sigmoid(output)
        loss = criterion(output, gts)
        loss_running += loss.data.item()
        loss.backward()
        optimizer.step()
    loss_list.append(loss_running)
    if epoch % print_every == 0:
        print('Epoch: %d, loss %f' % (epoch, loss_list[-1]))

print(loss_list)

save_dir_res = os.path.join(save_dir, 'Results', seq_name)
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)


with torch.no_grad():
    for i, batch in enumerate(testloader):
        inputs, gts, fname = batch['image'], batch['gt'], batch['fname']
        inputs, gts = inputs.to(device), gts.to(device)
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)

        for j in range(int(outputs.size()[0])):
            output_np = outputs[j].cpu().detach().numpy().copy()
            out_img = np.argmin(output_np, axis=1)
            # print(out_img.shape)
            print("saving seq %s.." % fname[j])
            plt.imsave(os.path.join(save_dir_res, os.path.basename(fname[j]) + '.png'), out_img)