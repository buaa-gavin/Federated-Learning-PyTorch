import torch
from PIL import Image
from torchvision import transforms

device = torch.device('cuda')
transform = transforms.Compose([
    transforms.Resize(28),  # 压缩成28*28
    transforms.CenterCrop(28),
    transforms.Grayscale(num_output_channels=1),  # RGB转灰度图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

def apply(img_path):
    net = torch.load('model.pkl')  # 加载模型
    # print(net)
    net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)  # 打开
    img = transform(img).unsqueeze(0)  # 变形
    # print(img.size())
    image = img.to(device)
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    classify = ['maligant', 'normal / benign']
    print('this image maybe:', classify[int(predicted[0])])

if __name__ == '__main__':
    apply('../image/normal (6).png')
