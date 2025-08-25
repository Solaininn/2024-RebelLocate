import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PilImage
import pandas as pd
import os

def create_cnn(num_classes):
    # Load pre-trained ResNet
    model = models.resnet18(pretrained=True)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final full layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Only fc params use gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def get_optimizer(model, lr=1e-3):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    return optimizer

class BuildingDataset(Dataset):
    def __init__(self, image_dir, label_df, transform=None):
        self.image_dir = image_dir
        self.label_df = label_df
        self.transform = transform
        self.image_files = []

        self.img_label_mapping = {row['img_name']: row['label'] for _, row in self.label_df.iterrows()}
        self.label_mapping = {label: idx for idx, label in enumerate(self.label_df['label'].unique())}

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    label = self.img_label_mapping.get(file, None)  # None if not found
                    if label is not None and label in self.label_mapping:
                        self.image_files.append(os.path.join(root, file))

        print(f"Found {len(self.image_files)} images in {image_dir}")  # Debugging line
        print(f"Label mapping: {self.label_mapping}")  # Debugging line

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = PilImage.open(image_path).convert('RGB')

        # Extract the image name from the file path and get the label
        label = self.img_label_mapping.get(image_name, None)

        # Skip images with invalid or missing labels
        if label is None or label not in self.label_mapping:
            return self.__getitem__((idx + 1) % len(self))  # Try the next item

        label_idx = self.label_mapping[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

def train_cnn(image_dir,labels_csv_path, epochs=10, batch_size=1, lr=1e-3):
    # Extract image data
    label_df = pd.read_csv(labels_csv_path)

    # Build Label map
    label_mapping = {label: idx for idx, label in enumerate(label_df['label'].unique())}
    num_classes = len(label_mapping)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = BuildingDataset(image_dir, label_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    device = torch.device('cpu')
    model = create_cnn(num_classes).to(device)
    optimizer = get_optimizer(model, lr)
    criterion = nn.CrossEntropyLoss()

    epoch_results = []

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        accumulation_steps = 4  # Number of steps to accumulate gradients

        for step, (images,labels) in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate gradients and update weights every `accumulation_steps` steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct / total

        epoch_results.append((epoch + 1, epoch_loss, epoch_accuracy))

        print (f'Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | accuracy: {epoch_accuracy:.4f} | Accuracy: {epoch_accuracy*100:.4f}%')
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')


    return model, label_mapping, epoch_results

def predict_image(model, image_path, label_mapping):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = PilImage.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    predicted_label = idx_to_label[predicted.item()]
    return predicted_label