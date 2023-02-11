import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import cv2
from flask import Flask, jsonify, request
from PIL import Image
from torch import nn


def main():
    parser = argparse.ArgumentParser(description="Model serving")
    parser.add_argument("--input_dir", help="Model directory")
    parser.add_argument("--output_dir", help="Meaningless directory")
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--model_version", help="Model version")

    args = parser.parse_args()

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

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

    def generate_model_state_dict_name(model_name, model_version):
        return f"{model_name}_{str(model_version)}.pth"

    def load_model(model_name, model_version):
        device = torch.device("cpu")
        model = CustomCNNModel()
        filepath = os.path.join(
            args.input_dir,
            generate_model_state_dict_name(model_name, model_version),
        )
        model.load_state_dict(torch.load(filepath))
        model.to(device)
        return model

    model = load_model(args.model_name, args.model_version)

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        img = Image.open(request.files["image"].stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (32, 32))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        model.eval()
        tensor = torch.from_numpy(img).type(torch.float32)
        output = model(tensor)
        _, pred = torch.max(output.data, 1)

        return jsonify(label=classes[pred[0]])

    app.run(host="0.0.0.0", port=9090)

    # TODO: refactoring
    return predict


if __name__ == "__main__":
    main()
