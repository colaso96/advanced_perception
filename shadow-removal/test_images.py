# (Bronte) Sihan Li, Cole Crescas 2023

import argparse
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics import ConfusionMatrix
from resnet import ResNet
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Test GTSRB images")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="checkpoints/model_35.pth",
    help="The path to the model checkpoint",
)

parser.add_argument(
    "--test_dir",
    type=str,
    default="./adv_img/GTSRB",
    help="The path to the test images",
)

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
test_dir = args.test_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ResNet().to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

# Define the test set
data_transforms = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
    ]
)
test_set = ImageFolder(root=test_dir, transform=data_transforms)

# Define the test loader
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
)

if __name__ == '__main__':
    # Random seed
    torch.manual_seed(0)

    cm = ConfusionMatrix(num_classes=43, task='multiclass')

    # Test the model
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = torch.argmax(output, dim=1)
        correct += pred.eq(target.view_as(pred)).cpu().sum()
        cm.update(pred, target)

    # Produce confusion matrix
    print('Confusion matrix:')
    print(cm.compute().numpy())
    # Plot confusion matrix
    confusion_matrix = cm.compute().numpy()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(43))
    plt.yticks(np.arange(43))
    plt.title('Confusion matrix')
    # Add numbers
    for i in range(43):
        for j in range(43):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center')
    plt.colorbar()
    # Save confusion matrix with padding
    plt.tight_layout()
    plt.gcf().set_size_inches(15, 15)
    plt.savefig(f'{args.test_dir}_confusion_matrix.png', dpi=500)

    print('Test accuracy: {:.2f}%'.format(100.0 * correct / len(test_loader.dataset)))
