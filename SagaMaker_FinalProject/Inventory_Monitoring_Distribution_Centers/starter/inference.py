import json
import logging
import sys
import io
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"

labels = [
    '1', '2', '3', '4', '5'
]


def input_fn(request_body, request_content_type):
    if request_content_type == "image/*":
        # Define the transformation for the input image
        image = Image.open(io.BytesIO(request_body))
        logger.info(f"Image size: {image.size}")
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # Resize images
                transforms.ToTensor(),  # Convert images to tensors
            ]
        )

        return transform(image)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(data, model):

    input_tensor = data.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            device = torch.device("cpu")
            model = model.to(device)
            input_data = input_tensor.to(device)
            model.eval()
            with torch.jit.optimized_execution(True):  # type: ignore
                output = model(input_data)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_data = input_tensor.to(device)
            model.eval()
            output = model(input_data)

    return output


def output_fn(predicted, accept):

    # Get the predicted class index
    predicted_idx = torch.argmax(
        predicted, dim=1
    ).item()  # Get the index of the max score

    # Format the output based on the requested content type
    if accept == "text/csv":
        # Format the output as CSV
        response = f"predicted_index,predicted_label\n{predicted_idx},{labels[int(predicted_idx)]}"
        return response, "text/csv"
    else:
        # Convert the prediction to a JSON response
        response = {
            "predicted_index": predicted_idx,
            "predicted_label": labels[
                int(predicted_idx)
            ],  # Assuming you have a global 'labels' list
        }
        return json.dumps(response), "application/json"


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def net():
    """
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    return model
