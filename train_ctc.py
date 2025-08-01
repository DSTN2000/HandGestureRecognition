import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mamba_ssm import Mamba2
from dataset.dataset import get_dataset, gestures
import pandas as pd
from typing import List, Tuple, Optional

class GestureDataset(Dataset):
    """
    Dataset class for continuous gesture recognition.
    Handles variable-length sequences and prepares targets for CTC loss.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        # Create gesture mapping - note that gesture 0 (Non-gesture) becomes blank token
        # So actual gesture classes are 1-13, and blank is represented implicitly in CTC
        self.num_classes = len(gestures) - 1  # Exclude "Non-gesture" since it becomes blank
        self.gesture_names = gestures[1:]  # Skip "Non-gesture"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get landmark sequence - shape: (n_frames, 63)
        lmk_seq = torch.FloatTensor(row['lmk_seq'])
        
        # Process gesture_id for CTC
        gesture_id = row['gesture_id']
        
        if gesture_id == 0:  # Non-gesture becomes empty target for CTC
            target = torch.LongTensor([])  # Empty sequence
        else:
            # Shift gesture IDs down by 1 since we removed gesture 0 (Non-gesture)
            target = torch.LongTensor([gesture_id - 1])
        
        return lmk_seq, target

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences and creates length tensors needed for CTC loss.
    """
    sequences, targets = zip(*batch)
    
    # Get sequence lengths before padding
    seq_lengths = torch.LongTensor([len(seq) for seq in sequences])
    target_lengths = torch.LongTensor([len(target) for target in targets])
    
    # Pad sequences to the same length
    max_seq_len = max(seq_lengths)
    feature_dim = sequences[0].shape[1]  # Should be 63
    
    padded_sequences = torch.zeros(len(sequences), max_seq_len, feature_dim)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    # Concatenate all targets for CTC loss
    concat_targets = torch.cat(targets) if any(len(t) > 0 for t in targets) else torch.LongTensor([])
    
    return padded_sequences, concat_targets, seq_lengths, target_lengths

class Mamba2GestureRecognizer(nn.Module):
    """
    Mamba2-based continuous gesture recognition model.
    
    The architecture consists of:
    1. Input projection to transform 63D landmarks to model dimension
    2. Mamba2 blocks for sequence modeling
    3. Output projection to gesture classes + blank token for CTC
    """
    
    def __init__(self, 
                 input_dim: int = 63,  # 21 landmarks * 3 coordinates
                 d_model: int = 256,   # Model dimension
                 num_classes: int = 13,  # 14 gestures - 1 (excluding Non-gesture)
                 num_layers: int = 4,   # Number of Mamba2 layers
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection: transform landmark features to model dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Stack of Mamba2 layers for sequence modeling
        # Mamba2 is particularly good at capturing long-range dependencies
        # which is crucial for gesture recognition where context matters
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model) for _ in range(num_layers)
        ])
        
        # Layer normalization between Mamba blocks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Final dropout before output
        self.final_dropout = nn.Dropout(dropout)
        
        # Output projection: map to gesture classes
        # Note: CTC automatically handles the blank token, so we only need
        # to output probabilities for actual gesture classes
        self.output_projection = nn.Linear(d_model, num_classes)
        
    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Actual sequence lengths (before padding)
            
        Returns:
            logits: Output logits of shape (seq_len, batch_size, num_classes)
                   (Transposed for CTC loss which expects T x N x C)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input landmarks to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Pass through Mamba2 layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)  # Mamba2 processes the sequence
            x = layer_norm(x + residual)  # Add residual connection and normalize
        
        # Apply final dropout
        x = self.final_dropout(x)
        
        # Project to output classes
        logits = self.output_projection(x)  # (batch_size, seq_len, num_classes)
        
        # Transpose for CTC loss: (seq_len, batch_size, num_classes)
        logits = logits.transpose(0, 1)
        
        return logits

class GestureTrainer:
    """
    Training class that handles the training loop, CTC loss computation,
    and model evaluation for gesture recognition.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """
        Train the model for one epoch and return sample outputs with accuracy.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        sample_output = None
        
        # Accuracy tracking
        total_samples = 0
        total_correct = 0
        
        # Randomly select a batch index to capture outputs from
        import random
        random_batch_idx = random.randint(0, len(dataloader) - 1)
        
        for batch_idx, (sequences, targets, seq_lengths, target_lengths) in enumerate(dataloader):
            # Move data to device
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            seq_lengths = seq_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(sequences, seq_lengths)
            
            # Apply log_softmax for CTC loss
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute CTC loss
            loss = self.ctc_loss(log_probs, targets, seq_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy for this batch
            with torch.no_grad():
                # Decode predictions
                predictions = self.decode_ctc_predictions(log_probs.detach().cpu(), seq_lengths.detach().cpu())
                
                # Decode targets for comparison
                target_start = 0
                decoded_targets = []
                for i, target_len in enumerate(target_lengths):
                    if target_len > 0:
                        target_seq = targets[target_start:target_start + target_len].tolist()
                        decoded_targets.append(target_seq)
                    else:
                        decoded_targets.append([])  # Empty target (Non-gesture)
                    target_start += target_len
                
                # Count exact matches
                for pred, target in zip(predictions, decoded_targets):
                    total_samples += 1
                    if pred == target:
                        total_correct += 1
            
            if batch_idx % 10 == 0:
                current_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Running Accuracy: {current_accuracy:.4f}')
            
            # Capture outputs for the randomly selected batch
            if batch_idx == random_batch_idx:
                # Store sample data for logging (move to CPU to avoid GPU memory issues)
                sample_output = {
                    'sequences': sequences.detach().cpu(),
                    'log_probs': log_probs.detach().cpu(),
                    'seq_lengths': seq_lengths.detach().cpu(),
                    'targets': targets.detach().cpu(),
                    'target_lengths': target_lengths.detach().cpu(),
                    'batch_idx': batch_idx
                }
        
        # Calculate final epoch accuracy
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, epoch_accuracy, sample_output

    
    def decode_ctc_predictions(self, log_probs, seq_lengths):
        """
        Decode CTC predictions using greedy decoding.
        This collapses repeated predictions and removes blank tokens.
        """
        predictions = []
        
        for i, length in enumerate(seq_lengths):
            # Get predictions for this sequence
            seq_preds = torch.argmax(log_probs[:length, i], dim=-1)
            
            # Greedy CTC decoding: collapse repeats and remove blanks
            decoded = []
            prev = None
            
            for pred in seq_preds:
                pred_item = pred.item()
                # Skip blanks (class 0) and repeated predictions
                if pred_item != 0 and pred_item != prev:
                    decoded.append(pred_item)
                prev = pred_item
            
            predictions.append(decoded)
        
        return predictions
    
    def log_sample_predictions(self, sample_output, epoch, gesture_names):
        """
        Log predictions for the sample batch.
        """
        if sample_output is None:
            print("No sample output captured this epoch.")
            return
            
        log_probs = sample_output['log_probs']
        seq_lengths = sample_output['seq_lengths']
        targets = sample_output['targets']
        target_lengths = sample_output['target_lengths']
        batch_idx = sample_output['batch_idx']
        
        print(f"\n--- Sample Predictions for Epoch {epoch} (Batch {batch_idx}) ---")
        
        # Decode predictions
        predictions = self.decode_ctc_predictions(log_probs, seq_lengths)
        
        # Decode targets for comparison
        target_start = 0
        decoded_targets = []
        for i, target_len in enumerate(target_lengths):
            if target_len > 0:
                target_seq = targets[target_start:target_start + target_len].tolist()
                decoded_targets.append(target_seq)
            else:
                decoded_targets.append([])  # Empty target (Non-gesture)
            target_start += target_len
        
        # Print predictions vs targets
        for i, (pred, target) in enumerate(zip(predictions, decoded_targets)):
            print(f"Sample {i+1}:")
            print(f"  Target gesture IDs: {target}")
            if target:
                target_names = [gesture_names[idx] for idx in target]
                print(f"  Target gestures: {target_names}")
            else:
                print(f"  Target gestures: ['Non-gesture']")
            
            print(f"  Predicted gesture IDs: {pred}")
            if pred:
                pred_names = [gesture_names[idx] for idx in pred]
                print(f"  Predicted gestures: {pred_names}")
            else:
                print(f"  Predicted gestures: ['Non-gesture']")
            
            # Calculate simple accuracy (exact match)
            accuracy = "\033[92m✓\033[0m" if pred == target else "\033[91m✗\033[0m"
            print(f"  Match: {accuracy}")
            print()

def create_model_and_trainer():
    """
    Factory function to create model and trainer with sensible defaults.
    """
    model = Mamba2GestureRecognizer(
        input_dim=63,          # 21 landmarks * 3 coordinates
        d_model=256,           # Model dimension - balance between capacity and efficiency
        num_classes=13,        # 14 total gestures - 1 (Non-gesture becomes blank)
        num_layers=4,          # Number of Mamba2 layers
        dropout=0.1            # Dropout for regularization
    )
    
    trainer = GestureTrainer(model)
    return model, trainer

def main():
    """
    Example training script showing how to use the model with sample logging.
    """
    # Load dataset
    print("Loading dataset...")
    df = get_dataset()
    print(f"\033[92mDataset loaded with {len(df)} samples\033[0m")
    
    # Create dataset and dataloader
    dataset = GestureDataset(df)
    dataloader = DataLoader(
        dataset, 
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Create model and trainer
    model, trainer = create_model_and_trainer()
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Get gesture names for logging (excluding Non-gesture)
    gesture_names = gestures[1:]  # Skip "Non-gesture" since it's the blank token
    
    # Training loop
    print("Starting training...")
    num_epochs = 200
    
    # Track best accuracy
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss, epoch_accuracy, sample_output = trainer.train_epoch(dataloader, optimizer, scheduler)
        
        # Print epoch results
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average accuracy: {epoch_accuracy:.4f} ({epoch_accuracy*100:.2f}%)")
        
        # Track best accuracy
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            print(f"New best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Log sample predictions at the end of each epoch
        trainer.log_sample_predictions(sample_output, epoch+1, gesture_names)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': epoch_accuracy,
            }, f'gesture_model_epoch_{epoch+1}.pth')
    
    print(f"\nTraining completed!")
    print(f"Best accuracy achieved: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")


if __name__ == '__main__':
    main()