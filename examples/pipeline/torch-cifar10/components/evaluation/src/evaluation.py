import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="Model tester")
    parser.add_argument("--input_dir", help="Model directory")
    parser.add_argument("--output_dir", help="Meaningless directory")
    parser.add_argument(
        "--preprocessed_data_dir", help="Preprocessed data directory"
    )
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--model_version", help="Model version")

    args = parser.parse_args()

    batch_size = 128

    class CustomCNNModel(nn.Module):
        """model class"""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def load_preprocessed_data():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test = torchvision.datasets.CIFAR10(
            root=args.preprocessed_data_dir,
            train=False,
            download=False,
            transform=transform,
        )
        return test

    def generate_dataloader(test_dataset, batch_size=100):
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return test_dataloader

    def generate_model_state_dict_name(model_name, model_version):
        return f"{model_name}_{str(model_version)}.pth"

    preprocessed_data = load_preprocessed_data()
    test_dataloader = generate_dataloader(
        preprocessed_data, batch_size=batch_size
    )

    device = torch.device("cpu")
    model = CustomCNNModel()

    filepath = os.path.join(
        args.input_dir,
        generate_model_state_dict_name(args.model_name, args.model_version),
    )

    model.load_state_dict(torch.load(filepath))
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(images.size())
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    print(f"Accuracy of the network on the 10000 test images: {accuracy} %")

    with open("/output.txt", "w", encoding="utf-8") as f:
        f.write(args.output_dir)

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
