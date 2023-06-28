import torch
import os

from catflow_worker.types import Prediction


class Model:
    def __init__(self, model_name, threshold):
        self.threshold = threshold
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=f"{model_name}.pt", trust_repo=True
        )
        self.model_name = os.path.basename(model_name)

    def predict(self, pil_image):
        """Run prediction on a PIL image"""

        # Perform inference
        results = self.model(pil_image)

        # Get bounding boxes and labels
        predictions = []
        for result in results.xywh[0].tolist():
            x, y, width, height, confidence, class_id = result
            predictions.append(
                Prediction(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    label=results.names[class_id],
                )
            )

        return predictions
