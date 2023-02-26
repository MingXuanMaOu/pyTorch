import torchvision
import torchvision.transforms as transforms
import pylab
import torch
from matplotlib import pylab as plt
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")
data_dir = "./fashion_mnist/"
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(data_dir,train=True,transform=transform,download=True)

print("训练数据集条数：",len(train_dataset))
val_dataset = torchvision.datasets.FashionMNIST(root=data_dir,train=False,transform=transform)
print(len(val_dataset))
im = train_dataset[0][0].numpy()
im = im.reshape(-1,28)
plt.imshow(im)
plt.show()
print("该图片的标签：",train_dataset[0][1])

batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

def imshow(img):
    print("图片形状：",np.shape(img))
    img = img / 2 + .5
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg,(1,2,0)))

classes = ('T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle_Boot')
sample = iter(train_loader)
images, labels = sample.__next__()
print('样本形状：',np.shape(images))
print('样本标签',labels)
imshow(torchvision.utils.make_grid(images,nrow=batch_size))
plt.show()
print(','.join('%5s' % classes[labels[j]] for j in range(len(images))))

