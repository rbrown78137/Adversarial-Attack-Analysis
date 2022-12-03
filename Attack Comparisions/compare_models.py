import torchvision.datasets
from torchvision.models import vgg16, resnet18,squeezenet1_0,alexnet,densenet201

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
model1 = resnet18(pretrained=True)
model1.eval()
model1.to(device)

model2 = vgg16(pretrained=True)
model2.eval()
model2.to(device)

model3 = squeezenet1_0(pretrained=True)
model3.eval()
model3.to(device)

model4 = alexnet(pretrained=True)
model4.eval()
model4.to(device)

model5 = densenet201(pretrained=True)
model5.eval()
model5.to(device)
# Sample Attack Method
attack_method1 = torchattacks.FGSM(model1, eps=2/255)
attack_method2 = torchattacks.FGSM(model2, eps=2/255)
attack_method3 = torchattacks.FGSM(model3, eps=2/255)
attack_method4 = torchattacks.FGSM(model4, eps=2/255)
attack_method5 = torchattacks.FGSM(model5, eps=2/255)

# Account For Normalization in Model
attack_method1.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method2.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method3.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method4.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attack_method5.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Loop Through Images and Generate Attacks
counter = 0
correct_predictions_model1 = 0
correct_predictions_model2 = 0
correct_predictions_model3 = 0
correct_predictions_model4 = 0
correct_predictions_model5 = 0
successful_attacks_model1 = 0
successful_attacks_model2 = 0
successful_attacks_model3 = 0
successful_attacks_model4 = 0
successful_attacks_model5 = 0

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
    predictions1 = model1(image)
    predictions1 = predictions1.argmax()

    predictions2 = model2(image)
    predictions2 = predictions2.argmax()

    predictions3 = model3(image)
    predictions3 = predictions3.argmax()

    predictions4 = model4(image)
    predictions4 = predictions4.argmax()

    predictions5 = model5(image)
    predictions5 = predictions5.argmax()

    # Analyze successful predictions of model
    if predictions1 == label:
        # Generate attack on original image
        adv_images = attack_method1(image, label)
        wrong_labels = model1(adv_images).argmax().unsqueeze(0)
        correct_predictions_model1 += 1
        if not predictions1 == wrong_labels:
            successful_attacks_model1 += 1

    if predictions2 == label:
        # Generate attack on original image
        adv_images = attack_method2(image, label)
        wrong_labels = model2(adv_images).argmax().unsqueeze(0)
        correct_predictions_model2 += 1
        if not predictions2 == wrong_labels:
            successful_attacks_model2 += 1

    if predictions3 == label:
        # Generate attack on original image
        adv_images = attack_method3(image, label)
        wrong_labels = model3(adv_images).argmax().unsqueeze(0)
        correct_predictions_model3 += 1
        if not predictions3 == wrong_labels:
            successful_attacks_model3 += 1

    if predictions4 == label:
        # Generate attack on original image
        adv_images = attack_method4(image, label)
        wrong_labels = model4(adv_images).argmax().unsqueeze(0)
        correct_predictions_model4 += 1
        if not predictions4 == wrong_labels:
            successful_attacks_model4 += 1

    if predictions5 == label:
        # Generate attack on original image
        adv_images = attack_method5(image, label)
        wrong_labels = model5(adv_images).argmax().unsqueeze(0)
        correct_predictions_model5 += 1
        if not predictions5 == wrong_labels:
            successful_attacks_model5 += 1

    if correct_predictions_model1 > 0:
        print(f"{successful_attacks_model1} out of {correct_predictions_model1} attacks successful: {successful_attacks_model1/correct_predictions_model1}")
    if correct_predictions_model2 > 0:
        print(f"{successful_attacks_model2} out of {correct_predictions_model2} attacks successful: {successful_attacks_model2/correct_predictions_model2}")
    if correct_predictions_model1 > 0:
        print(f"{successful_attacks_model3} out of {correct_predictions_model3} attacks successful: {successful_attacks_model3/correct_predictions_model3}")
    if correct_predictions_model1 > 0:
        print(f"{successful_attacks_model4} out of {correct_predictions_model4} attacks successful: {successful_attacks_model4/correct_predictions_model4}")
    if correct_predictions_model1 > 0:
        print(f"{successful_attacks_model5} out of {correct_predictions_model5} attacks successful: {successful_attacks_model5/correct_predictions_model5}")
