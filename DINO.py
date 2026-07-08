import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_DINO(): #used in the primary file to load DINO for use
    device = "cuda" if torch.cuda.is_available() else "cpu" #forces use of GPU if there is a GPU otherwise it uses CPU
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    print("DINOv2")
    print("Device", device)
    return device, model 

def setup():
    device, model = load_DINO()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        )
    ])
    return device, transform, model

def DINO_Vector(dataset_path):
    device, transform, model = setup()
    data = datasets.ImageFolder(dataset_path, transform = transform)
    loader = DataLoader(data, batch_size = 16, shuffle = False)

    embeddings_list = []
    labels_list = []
    paths = [path for path, label in data.samples]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            embeddings = model(images)

            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())
    X = torch.cat(embeddings_list, dim = 0)
    Y = torch.cat(labels_list, dim = 0)
    return X, Y, paths, data.classes