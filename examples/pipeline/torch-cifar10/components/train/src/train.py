from __future__ import print_function

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument("--input_dir", help="Processed data directory")
    parser.add_argument("--output_dir", help="Output model directory")
    parser.add_argument("--epochs", help="Number of training epochs")
    parser.add_argument("--model_name", help="Output model name")
    parser.add_argument("--model_version", help="Output model version")

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

        def _forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        forward = _forward

    def load_preprocessed_data(input_dir):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train = torchvision.datasets.CIFAR10(
            root=input_dir, train=True, download=False, transform=transform
        )
        return train

    def generate_dataloader(train_dataset, batch_size=100):
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return train_dataloader

    def generate_model_state_dict_name(model_name, model_version):
        return f"{model_name}_{str(model_version)}.pth"

    preprocessed_data = load_preprocessed_data(args.input_dir)
    train_dataloader = generate_dataloader(
        preprocessed_data, batch_size=batch_size
    )

    device = torch.device("cpu")
    model = CustomCNNModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(int(args.epochs)):
        avg_cost = 0

        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)  # pylint: disable=not-callable

            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            avg_cost += loss.item()

        avg_cost /= len(train_dataloader) / batch_size
        print(f"Model => [Epoch: {epoch+1:>4}] loss : {avg_cost:>.7f}")

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    filepath = os.path.join(
        args.output_dir,
        generate_model_state_dict_name(args.model_name, args.model_version),
    )
    torch.save(model.state_dict(), filepath)

    with open("/output.txt", "w", encoding="utf-8") as f:
        f.write(args.output_dir)

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print("Training Done")


if __name__ == "__main__":
    main()
