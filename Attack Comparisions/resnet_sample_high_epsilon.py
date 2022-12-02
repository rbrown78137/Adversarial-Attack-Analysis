import torchvision.datasets
from torchvision.models import resnet18
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
model = resnet18(pretrained=True)
model.eval()
model.to(device)

# Sample Attack Method
attack_method = torchattacks.PGD(model, eps=40/255, alpha=40/255, steps=4)

# Account For Normalization in Model
attack_method.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Loop Through Images and Generate Attacks
counter = 0
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
        # Generate attack on original image
        adv_images = attack_method(image, label)
        wrong_labels = model(adv_images).argmax().unsqueeze(0)

        # Clip image and test clipped image before normalization
        inverse_normalized_image = inverse_normalize_input(adv_images)
        clipped_image = torch.clip(inverse_normalized_image, min=0, max=1)
        normalized_clipped_image = normalize_input(clipped_image)
        clipped_image_prediction = model(normalized_clipped_image).argmax().unsqueeze(0)

        # Display images if attack is successful
        if not label == clipped_image_prediction:
            print("Successful Attack")
            displayable_base = inverse_normalize_input(image).squeeze(0).to("cpu")
            displayable_attack = clipped_image.squeeze(0).to("cpu")
            fig = plt.figure()
            sub_plot = fig.add_subplot(1, 2, 1)
            sub_plot.imshow(displayable_base.permute(1, 2, 0))
            plt.axis('off')
            sub_plot = fig.add_subplot(1, 2, 2)
            sub_plot.imshow(displayable_attack.permute(1, 2, 0))
            plt.axis('off')
            # fig.show()
            fig.savefig("./discrete_sample_images/"+str(counter)+".png")
