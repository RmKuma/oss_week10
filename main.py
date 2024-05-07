import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_dir)[:int(0.95 * len(os.listdir(class_dir)))]:
                image_path = os.path.join(class_dir, image_name)
                self.images.append((self.transform(Image.open(image_path).convert('RGB')), self.class_to_idx[cls]))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        return image, label

class CatDogTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_dir)[int(0.95 * len(os.listdir(class_dir))):]:
                image_path = os.path.join(class_dir, image_name)
                self.images.append((self.transform(Image.open(image_path).convert('RGB')), self.class_to_idx[cls]))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        return image, label


batch_size = 128
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = CatDogDataset(root_dir='./PetImages', transform=transform)
test_dataset = CatDogTestDataset(root_dir='./PetImages', transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))

        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 128, 3),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(2))

        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(128, 128, 3),
          nn.ReLU())

        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=128*24*24, out_features=2))

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = ImageClassifier()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        steps = 20
        if i % steps == steps-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / steps:.3f}')
            running_loss = 0.0

    test_acc = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        test_pred_labels = outputs.argmax(dim=1)
        test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
    test_acc = test_acc / len(testloader)

    print("Accuracy : {} %".format(test_acc))

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


