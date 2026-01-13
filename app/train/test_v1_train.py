import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from app.models.efficientnet_test import EfficientNetTest
from PIL import Image
import numpy as np


def create_synthetic_mini_dataset(num_images_per_class=50):
    """Kreiraj sintetičke slike za test trening (brzo, bez download-a)."""
    data_mini_dir = "data_mini"
    os.makedirs(f"{data_mini_dir}/AI", exist_ok=True)
    os.makedirs(f"{data_mini_dir}/Real", exist_ok=True)
    
    # Kreiraj 50 "AI" slika (bijele sa šumom)
    for i in range(num_images_per_class):
        img = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(f"{data_mini_dir}/AI/ai_test_{i}.jpg")
    
    # Kreiraj 50 "Real" slika (sive sa varijacijama)
    for i in range(num_images_per_class):
        img = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(f"{data_mini_dir}/Real/real_test_{i}.jpg")
    
    print(f"✅ Sintetički mini dataset spreman: {data_mini_dir} (100 slika)")
    return data_mini_dir


def train_mini_model():
    """Trening EfficientNet na mini datasetu."""
    # Transformacije (ImageNet standard)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset + DataLoader
    mini_dataset = ImageFolder("data_mini", transform=transform)
    loader = DataLoader(mini_dataset, batch_size=8, shuffle=True, num_workers=0)  # num_workers=0 za Windows
    
    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    model = EfficientNetTest(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 1 epoha za test
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Progress
            print(f"\rBatch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}", end="")
        
        avg_loss = running_loss / len(loader)
        print(f"\nEpoch {epoch+1} završena: Average Loss = {avg_loss:.4f}")
    
    # AUTOMATSKO kreiranje models foldera + spremanje
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/efficientnet_test_mini.pth")
    print("✅ Model spreman: models/efficientnet_test_mini.pth")


if __name__ == "__main__":
    # Kreiraj sintetički dataset + treniraj
    create_synthetic_mini_dataset(num_images_per_class=50)
    train_mini_model()
