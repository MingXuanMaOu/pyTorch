import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision.transforms as transforms
import sys

sys.stdout.reconfigure(encoding="utf-8")
myimg = mpimg.imread('img.jpg')
plt.imshow(myimg)
plt.axis('off')
plt.show()
print(myimg.shape)

pil2tensor = transforms.ToTensor()
rgb_image = pil2tensor(myimg)
print(rgb_image)
print(rgb_image.shape)

sobelfilter =  torch.tensor([[-1.0,0,1],  [-2,0,2],  [-1.0,0,1.0]]*3).reshape([1,3,3, 3])
print(sobelfilter)

op =torch.nn.functional.conv2d(rgb_image.unsqueeze(0), sobelfilter, stride=3,padding = 1) #3个通道输入，生成1个feature map
ret = (op - op.min()).div(op.max() - op.min())
print(ret)

plt.imshow(ret.squeeze(),cmap='Greys_r')
plt.axis('off')
plt.show()
