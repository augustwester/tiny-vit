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

batch_size = 64
num_epochs = 100

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

img_size = 32
patch_size = 4
num_patches = (img_size // patch_size) ** 2
patch_dim = 3 * patch_size ** 2
depth = 12

vit = ViT(depth, num_patches, patch_size, patch_dim, latent_dim=64, mlp_dim=512, num_classes=len(classes))
opt = Adam(vit.parameters(), lr=1e-4)
ce = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt.zero_grad()

        preds = vit(inputs)
        loss = ce(preds, labels)
        loss.backward()
        opt.step()
        print(loss)

print('Finished Training')