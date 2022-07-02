import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from vit import ViT

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 256
num_epochs = 512

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

img_size = 32
patch_size = 4
num_patches = (img_size // patch_size) ** 2
patch_dim = 3 * patch_size ** 2
depth = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vit = ViT(depth, num_patches, patch_size, patch_dim, embed_dim=64, mlp_dim=256, num_classes=10)
vit.to(device)

opt = Adam(vit.parameters(), lr=1e-4)
ce = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()

        preds = vit(inputs)
        loss = ce(preds, labels)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch}:", loss.item())
    if epoch % 10 != 0: continue
    num_correct = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        preds = vit(inputs).argmax(1)
        num_correct += len(labels) - torch.count_nonzero(labels - preds)
    print(f"Correct predictions: {num_correct}/{len(testset)}")
print('Finished Training')
