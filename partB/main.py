import os
import shutil
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

with tarfile.open("102flowers/102flowers.tgz", "r:gz") as tar:
    tar.extractall()

os.makedirs("flowers/class_0", exist_ok=True)

jpg_directory = "jpg"
image_files = os.listdir(jpg_directory)

for file in image_files:
    if file.endswith(".jpg"):
        src_path = os.path.join(jpg_directory, file)
        dst_path = os.path.join("flowers/class_0", file)
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)

dataset = ImageFolder(root="flowers", transform=transform)
dataset_size = len(dataset)
indices = list(range(dataset_size))
train_size = int(0.5 * dataset_size)
val_size = int(0.25 * dataset_size)
test_size = dataset_size - train_size - val_size

train_indices = np.random.choice(indices, size=train_size, replace=False)
remaining_indices = set(indices) - set(train_indices)
val_indices = np.random.choice(list(remaining_indices), size=val_size, replace=False)
test_indices = list(remaining_indices - set(val_indices))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


class YOLOv5(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(262144, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.model = torchvision.models.vgg19(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


yolov5 = YOLOv5(num_classes=102)
vgg19 = VGG19(num_classes=102)

yolov5 = yolov5.to(device)
vgg19 = vgg19.to(device)

criterion = nn.CrossEntropyLoss()
yolov5_optimizer = optim.Adam(yolov5.parameters(), lr=0.001)
vgg19_optimizer = optim.Adam(vgg19.parameters(), lr=0.001)


def fit(model, optimizer, criterion, train_loader, val_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    count = 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        for images, labels in train_loader:
            print(count)
            count+=1
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted_test = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted_test.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct_test / total_test

        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, val_losses, test_losses, train_accs, val_accs, test_accs



yolov5_train_losses, yolov5_val_losses, yolov5_test_losses, yolov5_train_accs, yolov5_val_accs, yolov5_test_accs = fit(yolov5, yolov5_optimizer, criterion, train_loader, val_loader, test_loader, num_epochs=10)


vgg19_train_losses, vgg19_val_losses, vgg19_test_losses, vgg19_train_accs, vgg19_val_accs, vgg19_test_accs = fit(vgg19, vgg19_optimizer, criterion, train_loader, val_loader, test_loader, num_epochs=10)



def plot_graphs(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs):
    epochs = len(train_losses)
    x = np.arange(1, epochs + 1)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses, label='validation Loss')
    plt.plot(x, test_losses, label='Test Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, train_accs, label='Train Accuracy')
    plt.plot(x, val_accs, label='Validation Accuracy')
    plt.plot(x, test_accs, label='Test Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_graphs(yolov5_train_losses, yolov5_val_losses, yolov5_test_losses, yolov5_train_accs, yolov5_val_accs, yolov5_test_accs)
plot_graphs(vgg19_train_losses, vgg19_val_losses, vgg19_test_losses, vgg19_train_accs, vgg19_val_accs, vgg19_test_accs)
