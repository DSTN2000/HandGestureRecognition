import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary as summary
import numpy as np
from mamba_ssm import Mamba2
from dataset.dataset import get_dataset, gestures
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from tqdm import tqdm
import random

import wandb
import wandb_workspaces.workspaces as ws
from utils.logging_callback import MyPyTorchLoggingCallback, create_workspace_sections

from model.model import Mamba2GestureRecognizer
from utils.collate_function import collate_fn
from utils.focal_loss import FocalLoss

class GestureDataset(Dataset):
    """
    Simplified dataset class for single gesture classification.
    Returns one label per sequence instead of continuous targets.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        # Keep all gestures including "Non-gesture" (gesture 0)
        self.num_classes = len(gestures)  # All 14 gesture classes
        self.gesture_names = gestures
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get landmark sequence - shape: (n_frames, 63)
        lmk_seq = torch.FloatTensor(row['lmk_seq'])
        
        target = torch.LongTensor([row['gesture_id']])
        
        return lmk_seq, target

class GestureTrainer:

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = FocalLoss()

    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """
        Train the model for one epoch with ```tqdm``` progress bar.
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Progress bar for training
        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_idx, (sequences, targets, seq_lengths) in enumerate(pbar):
            # Move data to device
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            seq_lengths = seq_lengths.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(sequences, seq_lengths)
            
            # Compute cross-entropy loss
            loss = self.loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Calculate accuracy for this batch
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                batch_correct = (predictions == targets).sum().item()
                batch_size = targets.size(0)
                
                total_correct += batch_correct
                total_samples += batch_size
                total_loss += loss.item()
                
                # Update progress bar
                current_acc = total_correct / total_samples
                current_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate final epoch metrics
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return avg_loss, epoch_accuracy
    
    def validate_epoch(self, dataloader, num_samples=5):
        """
        Validate the model and return sample predictions.
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        sample_predictions = []
        
        # Progress bar for validation
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, (sequences, targets, seq_lengths) in enumerate(pbar):
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                
                # Forward pass
                logits = self.model(sequences, seq_lengths)
                
                # Compute loss
                loss = self.loss_fn(logits, targets)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                batch_correct = (predictions == targets).sum().item()
                batch_size = targets.size(0)
                
                total_correct += batch_correct
                total_samples += batch_size
                total_loss += loss.item()
                
                # Collect sample predictions from first few batches
                if len(sample_predictions) < num_samples and batch_idx < 3:
                    probs = F.softmax(logits, dim=1)
                    for i in range(min(batch_size, num_samples - len(sample_predictions))):
                        sample_predictions.append({
                            'target': targets[i].item(),
                            'prediction': predictions[i].item(),
                            'confidence': probs[i, predictions[i]].item(),
                            'target_confidence': probs[i, targets[i]].item()
                        })
                
                # Update progress bar
                current_acc = total_correct / total_samples
                current_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate final validation metrics
        val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return avg_val_loss, val_accuracy, sample_predictions
    
    def log_sample_predictions(self, sample_predictions, epoch, gesture_names):
        """
        Log sample predictions from validation set.
        """
        if not sample_predictions:
            print("No sample predictions available.")
            return
            
        print(f"\n--- Validation Sample Predictions for Epoch {epoch} ---")
        
        for i, sample in enumerate(sample_predictions):
            target_id = sample['target']
            pred_id = sample['prediction']
            confidence = sample['confidence']
            target_conf = sample['target_confidence']
            
            print(f"Sample {i+1}:")
            print(f"  Target: {gesture_names[target_id]} (ID: {target_id}, Conf: {target_conf:.3f})")
            print(f"  Predicted: {gesture_names[pred_id]} (ID: {pred_id}, Conf: {confidence:.3f})")
            
            # Show accuracy
            accuracy = "\033[92m✓\033[0m" if pred_id == target_id else "\033[91m✗\033[0m"
            print(f"  Match: {accuracy}")
            print()


def create_model_and_trainer():
    """
    Factory function to create model and trainer with sensible defaults.
    """
    model = Mamba2GestureRecognizer(
        input_dim=63,          # 21 landmarks * 3 coordinates
        d_model=64,            # Model dimension
        num_classes=14,        # All 14 gesture classes
        num_layers=4,          # Number of Mamba2 layers
        dropout=0.1            # Dropout for regularization
    )
    
    trainer = GestureTrainer(model)
    return model, trainer


def main():
    # Load dataset
    print("Loading dataset...")
    full_df = get_dataset(threshold=0.5)
    print(f"\033[92mDataset loaded with {len(full_df)} samples\033[0m")
    
    # Split dataset into train and validation sets
    test_size = 0.2
    print("Splitting dataset into train/validation sets...")
    train_df, val_df = train_test_split(
        full_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=full_df['gesture_id']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Print class distribution
    print("\nClass distribution:")
    print("Training set:")
    print(train_df['gesture_id'].value_counts().sort_index())
    print("\nValidation set:")
    print(val_df['gesture_id'].value_counts().sort_index())
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(train_df)
    val_dataset = GestureDataset(val_df)
    batch_size=16

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=collate_fn
    )
    
    # Create model and trainer
    model, trainer = create_model_and_trainer()

    class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['gesture_id']),
    y=train_df['gesture_id']
    )
    class_weights = torch.FloatTensor(class_weights)

    trainer.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0) # Pass the weights to the focal loss as the alpha
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Get gesture names for logging
    gesture_names = gestures
    
    # Training loop with validation
    print("\nStarting training...")
    num_epochs = 200

    # Initialize wandb logging
    wandb.init(
        # set the wandb project where this run will be logged
        project="hgr",

        # track hyperparameters and run metadata with wandb.config
        config={
            "compile_config": {'loss': trainer.loss_fn, 'optimizer': optimizer},
            "epochs": num_epochs,
            "batch_size": batch_size,
            "model_summary": model,
            "validation_split": test_size,
            "dataset_size": len(train_dataset) + len(val_dataset), 
            "features": '63 features: raw coordinates',
        }
    )

    # Log model to wandb
    wandb.watch(model, log_freq=10)

    wandb.run.define_metric("*", step_metric="Epoch")

    # Create sections for each gesture class
    ws_url = f"{'/'.join(wandb.run.url.split('/')[:-2])}?nw=tkvgdf0zdae"
    workspace = ws.Workspace.from_url(ws_url)
    class_sections = create_workspace_sections(workspace, gesture_names)

    # Initialize logging callback
    logging_callback = MyPyTorchLoggingCallback(
        validation_dataloader=val_dataloader,
        class_names=gesture_names,
        model=model,
        device=trainer.device,
        workspace=workspace,
        class_sections=class_sections,
        log_frequency=1  # Log every epoch
    )

    # Set the name of the model
    model_name = wandb.run.name

    # Track best validation accuracy
    best_val_accuracy = 0.0
    
    # Main training loop with epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        # Training phase
        train_loss, train_accuracy = trainer.train_epoch(train_dataloader, optimizer, scheduler)
        
        # Validation phase
        val_loss, val_accuracy, sample_predictions = trainer.validate_epoch(val_dataloader, num_samples=5)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train_Loss': f'{train_loss:.4f}',
            'Train_Acc': f'{train_accuracy:.4f}',
            'Val_Loss': f'{val_loss:.4f}',
            'Val_Acc': f'{val_accuracy:.4f}'
        })
        
        # Log basic metrics to wandb
        wandb.log({
            'epoch/epoch': epoch,
            'epoch/loss': train_loss,
            'epoch/categorical_accuracy': train_accuracy,
            'epoch/val_loss': val_loss,
            'epoch/val_categorical_accuracy': val_accuracy,
            'epoch/learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Call the logging callback
        logging_callback.on_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy
        )

        # Print detailed epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  \033[92mNew best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)\033[0m")
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'full_model': model,
            }, f'./saved_models/best-{model_name}.pth')
        
        # Log sample predictions from validation set
        trainer.log_sample_predictions(sample_predictions, epoch+1, gesture_names)
        
    print(f"\nTraining completed!")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    # Log best model artifact to wandb
    wandb.save(f'./saved_models/best_{model_name}.pth')
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
