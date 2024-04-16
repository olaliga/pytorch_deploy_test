from torchvision.models import resnet50, ResNet50_Weights
import torch

# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = resnet50(weights="IMAGENET1K_V1")
input = torch.rand(2, 3, 224, 224)

print(model(input).shape)


