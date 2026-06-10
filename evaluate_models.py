import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image, ImageDraw
from torchvision import transforms
from models.deep_learning_models import MultimodalDrowsinessClassifier, LANDMARK_SET

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Image transforms matching server.py and Drowsiness_Detection.py
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SyntheticDrowsinessDataset(Dataset):
    """
    Simulates a sequence dataset of driver features.
    Length of sequence T = 30 frames.
    Class 0 = Alert
    Class 1 = Drowsy
    """
    def __init__(self, size=200, seq_len=30):
        self.size = size
        self.seq_len = seq_len
        self.frames = []
        self.landmarks = []
        self.labels = []
        
        for i in range(size):
            # Equal distribution of Alert and Drowsy classes
            label = 1 if i < size // 2 else 0
            
            # Generate frames sequence: (seq_len, 3, 64, 64)
            # Use PIL.ImageDraw to draw basic face crop shapes representing alert vs drowsy state
            frames_seq = []
            for t in range(seq_len):
                is_blinking = (label == 0 and 10 <= t <= 12)
                is_yawning = (label == 1)
                eyes_closed = (label == 1 or is_blinking)
                
                # Create a skin-toned background
                img = Image.new("RGB", (64, 64), color=(240, 200, 180))
                draw = ImageDraw.Draw(img)
                
                # Draw eyes
                if eyes_closed:
                    # Closed eyes: lines
                    draw.line([(18, 25), (26, 25)], fill=(0, 0, 0), width=2)
                    draw.line([(38, 25), (46, 25)], fill=(0, 0, 0), width=2)
                else:
                    # Open eyes: ellipses
                    draw.ellipse([(18, 22), (26, 28)], fill=(0, 0, 0))
                    draw.ellipse([(38, 22), (46, 28)], fill=(0, 0, 0))
                    
                # Draw mouth
                if is_yawning:
                    # Yawning: open vertical ellipse
                    draw.ellipse([(24, 38), (40, 52)], fill=(120, 0, 0))
                else:
                    # Closed mouth: horizontal line
                    draw.line([(24, 45), (40, 45)], fill=(150, 50, 50), width=2)
                    
                img_tensor = img_transforms(img)
                frames_seq.append(img_tensor)
                
            frames_seq_tensor = torch.stack(frames_seq)
            
            # Generate landmarks sequence: (seq_len, 21, 3) using a realistic face template
            lms_seq = np.zeros((seq_len, 21, 3))
            for t in range(seq_len):
                is_blinking = (label == 0 and 10 <= t <= 12)
                is_yawning = (label == 1)
                eyes_closed = (label == 1 or is_blinking)
                
                # Left eye (indices 0-5)
                lms_seq[t, 0] = [0.42, 0.45, 0.0]
                lms_seq[t, 3] = [0.48, 0.45, 0.0]
                if eyes_closed:
                    lms_seq[t, 1] = [0.44, 0.45, 0.0]
                    lms_seq[t, 2] = [0.46, 0.45, 0.0]
                    lms_seq[t, 4] = [0.46, 0.45, 0.0]
                    lms_seq[t, 5] = [0.44, 0.45, 0.0]
                else:
                    lms_seq[t, 1] = [0.44, 0.43, 0.0]
                    lms_seq[t, 2] = [0.46, 0.43, 0.0]
                    lms_seq[t, 4] = [0.46, 0.47, 0.0]
                    lms_seq[t, 5] = [0.44, 0.47, 0.0]
                    
                # Right eye (indices 6-11)
                lms_seq[t, 6] = [0.52, 0.45, 0.0]
                lms_seq[t, 9] = [0.58, 0.45, 0.0]
                if eyes_closed:
                    lms_seq[t, 7] = [0.54, 0.45, 0.0]
                    lms_seq[t, 8] = [0.56, 0.45, 0.0]
                    lms_seq[t, 10] = [0.56, 0.45, 0.0]
                    lms_seq[t, 11] = [0.54, 0.45, 0.0]
                else:
                    lms_seq[t, 7] = [0.54, 0.43, 0.0]
                    lms_seq[t, 8] = [0.56, 0.43, 0.0]
                    lms_seq[t, 10] = [0.56, 0.47, 0.0]
                    lms_seq[t, 11] = [0.54, 0.47, 0.0]
                    
                # Mouth (indices 12-17)
                lms_seq[t, 12] = [0.45, 0.58, 0.0]
                lms_seq[t, 15] = [0.55, 0.58, 0.0]
                if is_yawning:
                    lms_seq[t, 13] = [0.48, 0.53, 0.0]
                    lms_seq[t, 14] = [0.52, 0.53, 0.0]
                    lms_seq[t, 16] = [0.52, 0.63, 0.0]
                    lms_seq[t, 17] = [0.48, 0.63, 0.0]
                else:
                    lms_seq[t, 13] = [0.48, 0.57, 0.0]
                    lms_seq[t, 14] = [0.52, 0.57, 0.0]
                    lms_seq[t, 16] = [0.52, 0.59, 0.0]
                    lms_seq[t, 17] = [0.48, 0.59, 0.0]
                    
                # Nose tip (18), Left Cheek (19), Right Cheek (20)
                lms_seq[t, 18] = [0.50, 0.50, 0.0]
                lms_seq[t, 19] = [0.40, 0.50, 0.0]
                lms_seq[t, 20] = [0.60, 0.50, 0.0]
                
            # Add subtle noise to landmarks sequence
            lms_seq += np.random.randn(seq_len, 21, 3) * 0.002
            
            self.frames.append(frames_seq_tensor)
            self.landmarks.append(torch.tensor(lms_seq, dtype=torch.float32))
            self.labels.append(label)
            
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.frames[idx], self.landmarks[idx], self.labels[idx]

def train_and_evaluate():
    print("[INFO] Generating synthetic driver sequence dataset...")
    # Train set: 160 samples, Test set: 40 samples
    train_dataset = SyntheticDrowsinessDataset(size=160)
    test_dataset = SyntheticDrowsinessDataset(size=40)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("[INFO] Initializing Multimodal Drowsiness Classifier (CNN + TimeSformer + ST-GCN)...")
    model = MultimodalDrowsinessClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print(f"[INFO] Training model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for frames, landmarks, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(frames, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
        
    print("[INFO] Evaluating model on validation/testing set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, landmarks, labels in test_loader:
            outputs = model(frames, landmarks)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n==========================================")
    print(" EVALUATION RESULTS")
    print("==========================================")
    print(f" Accuracy  : {accuracy:.4f}")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" F1-Score  : {f1:.4f}")
    print("==========================================\n")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/drowsiness_model.pth")
    print("[INFO] Saved trained weights to 'models/drowsiness_model.pth'")
    
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "dataset_size": len(train_dataset) + len(test_dataset),
        "epochs_trained": epochs
    }
    
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("[INFO] Saved performance metrics to 'models/evaluation_results.json'")

if __name__ == "__main__":
    train_and_evaluate()
