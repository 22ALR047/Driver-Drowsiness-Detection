import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure parent and sibling directories are in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning_models import MultimodalDrowsinessClassifier
from evaluate_models import SyntheticDrowsinessDataset

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalDrowsinessClassifier().to(device)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'drowsiness_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded weights successfully.")
    else:
        print("No weights found.")
        return

    test_dataset = SyntheticDrowsinessDataset(size=40)
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
    
    frames, landmarks, labels = next(iter(test_loader))
    frames = frames.to(device)
    landmarks = landmarks.to(device)
    
    print("\n--- Testing under model.train() mode ---")
    model.train()
    with torch.no_grad():
        outputs = model(frames, landmarks)
        preds = torch.argmax(outputs, dim=1)
        print("Predictions :", preds.tolist())
        print("True Labels :", labels.tolist())
        
    print("\n--- Testing under model.eval() mode ---")
    model.eval()
    with torch.no_grad():
        outputs = model(frames, landmarks)
        preds = torch.argmax(outputs, dim=1)
        print("Predictions :", preds.tolist())
        print("True Labels :", labels.tolist())

if __name__ == '__main__':
    test()
