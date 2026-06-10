import torch
import torch.nn as nn
import torchvision.models as models

# LANDMARK CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# 21-point curated facial mesh landmark set:
# Left eye (6), Right eye (6), Mouth (6), Nose tip (1), Left cheek (1), Right cheek (1)
LANDMARK_SET = [362, 385, 387, 263, 373, 380, 33, 160, 158, 133, 153, 144, 61, 82, 312, 291, 317, 87, 1, 234, 454]

# ══════════════════════════════════════════════════════════════════════════════
# 1. CNN FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
class CNNFeatureExtractor(nn.Module):
    """Extracts spatial visual features from a face region crop using pre-trained MobileNetV3."""
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load pre-trained mobilenet_v3_small
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        
        # Freeze features parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(576, embedding_dim)
        
    def forward(self, x):
        # x shape: (B, 3, 64, 64)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ══════════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL TRANSFORMER (VideoMAE / TimeSformer inspired)
# ══════════════════════════════════════════════════════════════════════════════
class TemporalTransformer(nn.Module):
    """Processes temporal frame sequences using attention mechanism."""
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=2, output_dim=128):
        super().__init__()
        # PyTorch Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        out = self.transformer(x)
        # Average pooling over the sequence time dimension
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPATIAL-TEMPORAL GRAPH CONVOLUTIONAL NETWORK (ST-GCN)
# ══════════════════════════════════════════════════════════════════════════════
class GCNBlock(nn.Module):
    """Spatial Graph Convolution block."""
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.register_buffer('A', adj)
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        # x shape: (batch, seq_len, V, in_channels)
        # Symmetrically normalize the adjacency matrix A
        degree = torch.sum(self.A, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv = torch.diag(d_inv_sqrt)
        normalized_adj = torch.matmul(torch.matmul(D_inv, self.A), D_inv)
        
        # Spatial convolution: out = D^-0.5 * A * D^-0.5 * x * W
        out = torch.einsum('vw,btwc->btvc', normalized_adj, x)
        out = self.linear(out)
        return out

class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution block."""
    def __init__(self, in_channels, out_channels, adj, kernel_size=9, stride=1):
        super().__init__()
        self.gcn = GCNBlock(in_channels, out_channels, adj)
        # 1D Temporal Convolution along time dimension
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=((kernel_size - 1) // 2, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, V, in_channels)
        out = self.gcn(x)  # -> (batch, seq_len, V, out_channels)
        # Transpose to (batch, out_channels, seq_len, V) for 2D convolution filters
        out = out.permute(0, 3, 1, 2)
        out = self.tcn(out)
        # Permute back to (batch, seq_len, V, out_channels)
        out = out.permute(0, 2, 3, 1)
        return out

def normalize_landmarks(landmarks_tensor):
    """
    Normalizes a tensor of facial landmarks to be translation- and scale-invariant.
    Input shape: (batch, seq_len, V, 3) where V = 21 landmarks.
    """
    # Landmarks indices in LANDMARK_SET:
    # Nose tip: 18 (MediaPipe index 1)
    # Left Cheek: 19 (MediaPipe index 234)
    # Right Cheek: 20 (MediaPipe index 454)
    nose = landmarks_tensor[..., 18, :]  # Shape (..., 3)
    left = landmarks_tensor[..., 19, :]
    right = landmarks_tensor[..., 20, :]
    
    # Calculate horizontal span (cheek distance) to scale coordinates
    span = right[..., 0] - left[..., 0]
    span = torch.clamp(span, min=1e-5)
    
    # Subtract nose coordinates to center landmarks and divide by span to scale
    nose_expanded = nose.unsqueeze(-2)  # Shape (..., 1, 3)
    span_expanded = span.unsqueeze(-1).unsqueeze(-1)  # Shape (..., 1, 1)
    
    return (landmarks_tensor - nose_expanded) / span_expanded

class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for tracking facial landmarks."""
    def __init__(self, num_nodes=21, in_channels=3, output_dim=128):
        super().__init__()
        # Construct graph adjacency matrix based on facial structure
        adj = torch.zeros(num_nodes, num_nodes)
        
        # Self-loops
        for i in range(num_nodes):
            adj[i, i] = 1.0
            
        # Left eye contours (indices 0-5)
        for i in range(6):
            adj[i, (i + 1) % 6] = 1.0
            adj[(i + 1) % 6, i] = 1.0
            
        # Right eye contours (indices 6-11)
        for i in range(6):
            adj[6 + i, 6 + (i + 1) % 6] = 1.0
            adj[6 + (i + 1) % 6, 6 + i] = 1.0
            
        # Mouth contours (indices 12-17)
        for i in range(6):
            adj[12 + i, 12 + (i + 1) % 6] = 1.0
            adj[12 + (i + 1) % 6, 12 + i] = 1.0
            
        # Nose (18), Left Cheek (19), Right Cheek (20)
        adj[18, 19] = adj[19, 18] = 1.0
        adj[18, 20] = adj[20, 18] = 1.0
        adj[19, 20] = adj[20, 19] = 1.0
        
        # Connect features to nose to create a unified face graph
        adj[18, 0] = adj[0, 18] = 1.0
        adj[18, 6] = adj[6, 18] = 1.0
        adj[18, 12] = adj[12, 18] = 1.0
        
        self.block1 = STGCNBlock(in_channels, 32, adj)
        self.block2 = STGCNBlock(32, 64, adj)
        self.fc = nn.Linear(64 * num_nodes, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, V, in_channels)
        x = normalize_landmarks(x)
        out = self.block1(x)
        out = self.block2(out)  # (batch, seq_len, V, 64)
        # Average pool over temporal sequence
        out = torch.mean(out, dim=1)  # (batch, V, 64)
        # Flatten nodes and linear project
        out = out.view(out.size(0), -1)  # (batch, V * 64)
        out = self.fc(out)
        return out

# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTIMODAL DROWSINESS CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class MultimodalDrowsinessClassifier(nn.Module):
    """Fuses CNN-Transformer video features with landmark ST-GCN features."""
    def __init__(self, visual_dim=128, landmark_dim=128, num_classes=2):
        super().__init__()
        self.visual_extractor = CNNFeatureExtractor(embedding_dim=visual_dim)
        self.temporal_transformer = TemporalTransformer(embedding_dim=visual_dim, output_dim=visual_dim)
        self.landmark_stgcn = STGCN(in_channels=3, output_dim=landmark_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + landmark_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def train(self, mode=True):
        super().train(mode)
        # Lock visual feature extractor pre-trained layers in eval mode to prevent running stats corruption
        self.visual_extractor.features.eval()
        return self
        
    def forward(self, frames, landmarks):
        # frames shape: (batch, seq_len, C, H, W)
        # landmarks shape: (batch, seq_len, V, 3)
        batch_size, seq_len, C, H, W = frames.size()
        
        # 1. Flatten temporal dimension to extract spatial visual features frame-by-frame
        flat_frames = frames.view(batch_size * seq_len, C, H, W)
        frame_features = self.visual_extractor(flat_frames)  # (batch * seq_len, visual_dim)
        frame_features = frame_features.view(batch_size, seq_len, -1)  # (batch, seq_len, visual_dim)
        
        # 2. Model temporal visual dynamics
        temporal_features = self.temporal_transformer(frame_features)  # (batch, visual_dim)
        
        # 3. Model spatial-temporal landmark graph movements
        landmark_features = self.landmark_stgcn(landmarks)  # (batch, landmark_dim)
        
        # 4. Concatenate and classify
        fused_features = torch.cat([temporal_features, landmark_features], dim=1)  # (batch, visual_dim + landmark_dim)
        logits = self.classifier(fused_features)
        return logits
