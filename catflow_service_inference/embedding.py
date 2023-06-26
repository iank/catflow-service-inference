import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


# This is likely derived from https://github.com/christiansafka/img2vec
class ImageFeatureExtractor:
    def __init__(self):
        # Load the pretrained ResNet-18 model
        self.model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")

        # Get the average pooling layer from the ResNet-18 model
        self.layer = self.model._modules.get("avgpool")

        self.model.eval()

        # Define the image transformations
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def get_vector(self, image):
        t_img = Variable(
            self.normalize(self.to_tensor(self.scaler(image))).unsqueeze(0)
        )

        my_embedding = None

        def copy_data(m, i, o):
            nonlocal my_embedding
            my_embedding = o.data.reshape(o.data.size(1))

        h = self.layer.register_forward_hook(copy_data)

        self.model(t_img)

        h.remove()

        return my_embedding.tolist()
