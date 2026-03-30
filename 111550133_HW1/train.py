import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = './data/train'  # Please adjust this path before running
    batch_size = 64
    num_epochs = 80
    learning_rate = 5e-5
    
    os.makedirs("./checkpoints", exist_ok=True)

    # Data Augmentation (448px)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Model Definition (ResNeXt-101 backbone with modified classifier)
    model = models.resnext101_32x8d(weights="DEFAULT")
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 100)
    )
    model = model.to(device)

    # Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()

        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:
            save_path = f"./checkpoints/RESNEXT_448_E{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'idx_to_class': idx_to_class
            }, save_path)

if __name__ == '__main__':
    main()