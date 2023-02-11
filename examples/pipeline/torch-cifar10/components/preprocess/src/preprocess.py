import argparse
import os

import torchvision
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="Data processor")
    parser.add_argument("--input_dir", help="Raw data directory")
    parser.add_argument("--output_dir", help="Processed data directory")

    args = parser.parse_args()

    def save_data(output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        torchvision.datasets.CIFAR10(
            root=output_dir, train=True, download=True, transform=transform
        )
        torchvision.datasets.CIFAR10(
            root=output_dir, train=False, download=True, transform=transform
        )

    save_data(args.output_dir)

    with open("/output.txt", "w", encoding="utf-8") as f:
        f.write(args.output_dir)

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
