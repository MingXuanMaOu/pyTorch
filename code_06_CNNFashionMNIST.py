import torchvision
import torchvision.transforms as transforms
import pylab
import torch
from matplotlib import pylab as plt
import numpy as np
import sys
import torch.nn.functional as F

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

classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')
sample = iter(train_loader)
images, labels = sample.__next__()
print('样本形状：',np.shape(images))
print('样本标签',labels)
imshow(torchvision.utils.make_grid(images,nrow=batch_size))
plt.show()
print(','.join('%5s' % classes[labels[j]] for j in range(len(images))))

class myConNet(torch.nn.Module):
    def __init__(self):
        super(myConNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=12 * 4 * 4,out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120,out_features=60)
        self.out = torch.nn.Linear(in_features=60,out_features=10)

    def forward(self,t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2,stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2,stride=2)

        t = t.reshape(-1,12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        return t

if __name__ == '__main__':
    network = myConNet()
    print(network)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(),lr = 0.01)

    for epoch in range(2):
        running_loss = 0.0
        for i,data in enumerate(train_loader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    torch.save(network.state_dict(),'./CNNFashionMNIST.pth')

    network.load_state_dict(torch.load('./CNNFashionMNIST.pth'))
    dataiter = iter(test_loader)
    images, labels = dataiter.__next__()
    inputs, labels = images.to(device), labels.to(device)

    imshow(torchvision.utils.make_grid(images,nrow=batch_size))
    print('真实标签：',' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    outputs = network(inputs)
    _, predicted = torch.max(outputs,1)
    print('预测结果：',' '.join('%5s' % classes[predicted[j]] for j in range(len(images))))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs, labels = images.to(device), labels.to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    sumacc = 0
    for i in range(10):
        Accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy))
        sumacc = sumacc + Accuracy
    print('Accuracy of all : %2d %%' % (sumacc / 10.))






