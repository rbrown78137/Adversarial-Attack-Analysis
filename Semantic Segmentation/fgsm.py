import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dice_loss import DiceLoss
from dataset import CustomImageDataset
from bayesian_model import BayesianModel
class FGSM:
    def __init__(self):
        self.loss_function = DiceLoss()
    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 255)
        # Return the perturbed image
        return perturbed_image


    def attack_model(self, model, device, data,target, epsilon ):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # Calculate the loss
        loss = self.loss_function(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        # # Re-classify the perturbed image
        # output = model(perturbed_data)

        # Return the accuracy and an adversarial example
        return perturbed_data


if __name__ == "__main__":
    dataset = CustomImageDataset()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = BayesianModel()
    model.load_state_dict(torch.load("saved_models/train_network.pth"))
    model.eval()
    model.to("cuda")
    fgsm = FGSM()
    counter = 0
    for images, labels in test_loader:
        counter+=1
        images = images.to("cuda")
        labels = labels.to("cuda")
        new_image = fgsm.attack_model(model,"cuda",images, labels, 8)
        image_to_display = images[0].permute(1, 2, 0).to("cpu").detach().to(dtype=torch.long)
        new_image_to_display = new_image[0].permute(1, 2, 0).to("cpu").detach().to(dtype=torch.long)
        prediction = model(images.detach()).argmax(1).squeeze(0).to("cpu")
        new_prediction = model(new_image.detach()).argmax(1).squeeze(0).to("cpu")
        # plt.imshow(image_to_display)
        # plt.imshow(new_image_to_display)
        # plt.imshow(prediction)
        # plt.imshow(new_prediction)
        fig = plt.figure()
        sub_plot = fig.add_subplot(2, 2, 1)
        sub_plot.imshow(image_to_display)
        plt.axis('off')
        sub_plot = fig.add_subplot(2, 2, 2)
        sub_plot.imshow(prediction)
        plt.axis('off')
        sub_plot = fig.add_subplot(2, 2, 3)
        sub_plot.imshow(new_image_to_display)
        plt.axis('off')
        sub_plot = fig.add_subplot(2, 2, 4)
        sub_plot.imshow(new_prediction)
        plt.axis('off')
        # fig.show()
        fig.savefig("./attack_images/" + str(counter) + ".png")
        breakpointpos = 1
