import torchvision.datasets
from torchvision.models import densenet201
import time
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import constants
import torchattacks

device = "cuda" if torch.cuda.is_available() else "cpu"

# PIL Image to Normalized Tensor
image_net_transformer = transforms.Compose([
        transforms.Resize(
            (299, 299)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])

# Normalized Tensor to Viewable Tensor
inverse_normalize_input = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# Viewable Tensor to Normalized Tensor
normalize_input = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])
# Load ImageNet Files
image_net_validation_set = torchvision.datasets.ImageNet(root=constants.path_to_image_net, split="val")

# Load Model
model = densenet201(pretrained=True)
model.eval()
model.to(device)

# Attack Methods
attack_method1 = torchattacks.FGSM(model, eps=1/255)
attack_method2 = torchattacks.FGSM(model, eps=2/255)
attack_method3 = torchattacks.FGSM(model, eps=8/255)
attack_method4 = torchattacks.FGSM(model, eps=16/255)
attack_method5 = torchattacks.FGSM(model, eps=32/255)

# Account For Normalization in Model
attack_method1.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method2.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method3.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method4.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method5.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Loop Through Images and Generate Attacks
counter = 0
successes_attack_method1 = 0
successes_attack_method2 = 0
successes_attack_method3 = 0
successes_attack_method4 = 0
successes_attack_method5 = 0
successful_predictions = 0
for image_pil, label_input in image_net_validation_set:
    print(f"Loading image{counter}")
    counter += 1

    # Generate input image to model
    image_input = image_net_transformer(image_pil)
    image = torch.zeros(1, image_input.shape[0], image_input.shape[1], image_input.shape[2])
    image[..., :, :, :] = image_input
    image = image.to(device)

    # Generate input label to model
    label = torch.zeros(1).to(device).to(dtype=torch.long)
    label[0] = label_input

    # Get prediction of image
    predictions = model(image)
    predictions = predictions.argmax()

    # Analyze successful predictions of model
    if predictions == label:
        predictions = predictions.unsqueeze(0)
        successful_predictions += 1
        # Generate attack on original image
        attack_image1 = attack_method1(image, label)
        wrong_label1 = model(attack_image1).argmax().unsqueeze(0)
        if not predictions == wrong_label1:
            successes_attack_method1 += 1

        attack_image2 = attack_method2(image, label)
        wrong_label2 = model(attack_image2).argmax().unsqueeze(0)
        if not predictions == wrong_label2:
            successes_attack_method2 += 1

        attack_image3 = attack_method3(image, label)
        wrong_label3 = model(attack_image3).argmax().unsqueeze(0)
        if not predictions == wrong_label3:
            successes_attack_method3 += 1

        attack_image4 = attack_method4(image, label)
        wrong_label4 = model(attack_image4).argmax().unsqueeze(0)
        if not predictions == wrong_label4:
            successes_attack_method4 += 1

        attack_image5 = attack_method5(image, label)
        wrong_label5 = model(attack_image5).argmax().unsqueeze(0)
        if not predictions == wrong_label5:
            successes_attack_method5 += 1
        print(f"Success rate of with FGSM epsilon   1/255: {successes_attack_method1 / successful_predictions}")
        print(f"Success rate of with FGSM epsilon   2/255: {successes_attack_method2 / successful_predictions}")
        print(f"Success rate of with FGSM epsilon   8/255: {successes_attack_method3 / successful_predictions}")
        print(f"Success rate of with FGSM epsilon  16/255: {successes_attack_method4 / successful_predictions}")
        print(f"Success rate of with FGSM epsilon  32/255: {successes_attack_method5 / successful_predictions}")

print(f"Success rate of with FGSM epsilon   1/255: {successes_attack_method1 / successful_predictions}")
print(f"Success rate of with FGSM epsilon   2/255: {successes_attack_method2 / successful_predictions}")
print(f"Success rate of with FGSM epsilon   8/255: {successes_attack_method3 / successful_predictions}")
print(f"Success rate of with FGSM epsilon  16/255: {successes_attack_method4 / successful_predictions}")
print(f"Success rate of with FGSM epsilon  32/255: {successes_attack_method5 / successful_predictions}")
