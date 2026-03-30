import os
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = './data/test'  # Please adjust this path before running
    checkpoint_path = "./checkpoints/RESNEXT_448_BEST.pth" # Adjust path to your .pth file
    output_csv = "prediction.csv"

    print("Loading model for inference...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    idx_to_class = ckpt['idx_to_class']

    # Initialize model with the exact same structure
    model = models.resnext101_32x8d()
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3), 
        torch.nn.Linear(model.fc.in_features, 100)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    # Inference Transforms (448px)
    test_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_names = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    results = []

    print("Starting inference...")
    with torch.no_grad():
        for name in tqdm(img_names):
            img_path = os.path.join(test_dir, name)
            img = Image.open(img_path).convert("RGB")
            tensor = test_transforms(img).unsqueeze(0).to(device)
            
            # Test-Time Augmentation (TTA)
            out = torch.softmax(model(tensor), dim=1)
            out_f = torch.softmax(model(torch.flip(tensor, [3])), dim=1)
            avg_prob = (out + out_f) / 2
            
            _, pred = torch.max(avg_prob, 1)
            results.append({
                "image_name": name.split('.')[0], 
                "pred_label": idx_to_class[pred.item()]
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Inference completed. Results saved to {output_csv}.")

if __name__ == '__main__':
    main()