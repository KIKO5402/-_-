import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from timm import create_model
import numpy as np
import pandas as pd
import time
import scipy

def append_results_to_dataframe(results_df, epoch, train_loss, test_loss, throughput, accuracy):
    new_row = pd.DataFrame({
        'Epoch': [epoch],
        'Train Loss': [train_loss],
        'Test Loss': [test_loss],
        'Throughput': [throughput],
        'Accuracy': [accuracy]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Test Loss', 'Accuracy'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),  # Data Augmentation
    transforms.RandomRotation(10),      # Data Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


#train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

#train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
#test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


train_dataset = torch.utils.data.Subset(train_dataset, list(range(10000)))
test_dataset = torch.utils.data.Subset(test_dataset, list(range(10000)))


train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda")
#model = create_model('sequencer2d_s', pretrained=False, num_classes=10).to(device)
#model = create_model('efficientnet_b0', pretrained=False, num_classes=10).to(device)
model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    end_time = time.time()
    epoch_time = end_time - start_time
    throughput = len(train_loader.dataset) / epoch_time

    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    test_losses.append(test_loss / len(test_loader))
    accuracy = correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Throughput: {throughput:.2f} samples/sec, Accuracy: {accuracy:.4f}')
    results_df = append_results_to_dataframe(results_df, epoch + 1, train_losses[-1], test_losses[-1], throughput, accuracy)

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r-', label='Train Loss')
plt.plot(epochs, test_losses, 'b-', label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, 'g-', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

print(results_df)

input("Press Any Key to Exit ")