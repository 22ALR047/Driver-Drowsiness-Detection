import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.deep_learning_models import MultimodalDrowsinessClassifier, LANDMARK_SET

# Image transforms matching server.py and Drowsiness_Detection.py
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class NoNoiseSyntheticDrowsinessDataset(Dataset):
    def __init__(self, size=200, seq_len=30):
        self.size = size
        self.seq_len = seq_len
        self.frames = []
        self.landmarks = []
        self.labels = []
        
        for i in range(size):
            label = 1 if i < size // 2 else 0
            
            frames_seq = []
            for t in range(seq_len):
                is_blinking = (label == 0 and 10 <= t <= 12)
                is_yawning = (label == 1)
                eyes_closed = (label == 1 or is_blinking)
                
                img = Image.new("RGB", (64, 64), color=(240, 200, 180))
                draw = ImageDraw.Draw(img)
                
                if eyes_closed:
                    draw.line([(18, 25), (26, 25)], fill=(0, 0, 0), width=2)
                    draw.line([(38, 25), (46, 25)], fill=(0, 0, 0), width=2)
                else:
                    draw.ellipse([(18, 22), (26, 28)], fill=(0, 0, 0))
                    draw.ellipse([(38, 22), (46, 28)], fill=(0, 0, 0))
                    
                if is_yawning:
                    draw.ellipse([(24, 38), (40, 52)], fill=(120, 0, 0))
                else:
                    draw.line([(24, 45), (40, 45)], fill=(150, 50, 50), width=2)
                    
                img_tensor = img_transforms(img)
                frames_seq.append(img_tensor)
                
            frames_seq_tensor = torch.stack(frames_seq)
            
            lms_seq = np.zeros((seq_len, 21, 3))
            for t in range(seq_len):
                is_blinking = (label == 0 and 10 <= t <= 12)
                is_yawning = (label == 1)
                eyes_closed = (label == 1 or is_blinking)
                
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
                    
                lms_seq[t, 18] = [0.50, 0.50, 0.0]
                lms_seq[t, 19] = [0.40, 0.50, 0.0]
                lms_seq[t, 20] = [0.60, 0.50, 0.0]
                
            self.frames.append(frames_seq_tensor)
            self.landmarks.append(torch.tensor(lms_seq, dtype=torch.float32))
            self.labels.append(label)
            
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.frames[idx], self.landmarks[idx], self.labels[idx]

def test():
    torch.manual_seed(42)
    train_dataset = NoNoiseSyntheticDrowsinessDataset(size=200)
    test_dataset = NoNoiseSyntheticDrowsinessDataset(size=50)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = MultimodalDrowsinessClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(15):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for frames, landmarks, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(frames, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        # Eval
        # Eval
        model.eval()
        train_eval_preds = []
        train_eval_labels = []
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for frames, landmarks, labels in train_loader:
                outputs = model(frames, landmarks)
                preds = torch.argmax(outputs, dim=1)
                train_eval_preds.extend(preds.cpu().numpy())
                train_eval_labels.extend(labels.cpu().numpy())
                
            for frames, landmarks, labels in test_loader:
                outputs = model(frames, landmarks)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        train_eval_acc = accuracy_score(train_eval_labels, train_eval_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc (during train): {correct/total:.4f} | Train Acc (during eval): {train_eval_acc:.4f} | Test Acc (during eval): {test_acc:.4f}")
        print(f"  Train Eval Preds: {train_eval_preds[:20]}")
        print(f"  Test Eval Preds : {test_preds[:20]}")

if __name__ == '__main__':
    test()
