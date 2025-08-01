from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr


class MyPyTorchLoggingCallback:
    def __init__(self, validation_dataloader, class_names, model, device, 
                 workspace: ws.Workspace, class_sections: Dict[str, ws.Section], 
                 log_frequency=1):
        self.validation_dataloader = validation_dataloader
        self.class_names = class_names
        self.model = model
        self.device = device
        self.workspace = workspace
        self.class_sections = class_sections
        self.log_frequency = log_frequency

    def _get_predictions(self):
        self.model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for sequences, targets, seq_lengths in tqdm(self.validation_dataloader, 
                                                       desc="Getting predictions", 
                                                       leave=False):
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                
                # Forward pass
                logits = self.model(sequences, seq_lengths)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(targets.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred)

    def _log_bar_charts(self, precision, recall, f1, epoch):
        """Create and log bar charts for per-class metrics"""
        # Create bar chart data
        precision_data = [[name, prec] 
                         for name, prec in zip(self.class_names, precision)]
        recall_data = [[name, rec] 
                      for name, rec in zip(self.class_names, recall)]
        f1_data = [[name, f1_score] 
                  for name, f1_score in zip(self.class_names, f1)]

        # Log as wandb bar charts
        wandb.log({
            "precision_per_class": wandb.plot.bar(
                wandb.Table(data=precision_data, columns=["Class", "Precision"]),
                "Class", "Precision", title="Precision per Class"
            ),
            "recall_per_class": wandb.plot.bar(
                wandb.Table(data=recall_data, columns=["Class", "Recall"]),
                "Class", "Recall", title="Recall per Class"
            ),
            "f1_per_class": wandb.plot.bar(
                wandb.Table(data=f1_data, columns=["Class", "F1"]),
                "Class", "F1", title="F1 Score per Class"
            ),
        })

    def on_epoch_end(self, epoch, train_loss=None, train_accuracy=None, 
                     val_loss=None, val_accuracy=None):
        """Call this method at the end of each epoch"""
        if epoch % self.log_frequency == 0:
            # Get predictions
            y_true, y_pred = self._get_predictions()

            # Calculate per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            # Log per-class metrics
            metrics_dict = {
                'Epoch': epoch,
                'epoch/loss': train_loss,
                'epoch/categorical_accuracy': train_accuracy,
                'epoch/val_loss': val_loss,
                'epoch/val_categorical_accuracy': val_accuracy,
            }
            
            for i, class_name in enumerate(self.class_names):
                metrics_dict[f'Precision for {class_name}'] = precision[i]
                metrics_dict[f'Recall for {class_name}'] = recall[i]
                metrics_dict[f'F1-Score for {class_name}'] = f1[i]
                metrics_dict[f'Support for {class_name}'] = support[i]

            wandb.log(metrics_dict)

            # Initialize workspace sections on first epoch
            section_metrics = ['Precision', 'Recall', 'F1-Score', 'Support']
            
            if epoch == 0 and not any([self.class_sections[class_name].panels 
                                     for class_name in self.class_names]):
                for i, class_name in enumerate(self.class_names):
                    # Log class-specific metrics to respective sections
                    self.class_sections[class_name].panels.clear()
                    for metric in section_metrics:
                        self.class_sections[class_name].panels.append(
                            wr.LinePlot(
                                title=metric, 
                                x='epoch', 
                                y=[f'{metric} for {class_name.replace(" ", "_")}']
                            )
                        )
            
            # Also log as custom bar chart
            self._log_bar_charts(precision, recall, f1, epoch)
            self.workspace.save()


def create_workspace_sections(workspace: ws.Workspace, class_names):
    """Create workspace sections for each gesture class"""
    try:
        existing_section_names = [section.name for section in workspace.sections]
        
        if any([class_name in existing_section_names for class_name in class_names]):
            print('Gesture sections already exist! Skipping...')
            class_sections = {}
            for section in workspace.sections:
                if section.name in class_names:
                    class_sections[section.name] = section
            return class_sections
            
        class_sections = {}
        for class_name in class_names:
            # Create a section for each class
            section = workspace.sections.append(ws.Section(
                name=f"{class_name}",
                is_open=True,
            ))
            class_sections[class_name] = section
            print(f"Created workspace section for class: {class_name}")
        
        workspace.save()
        return class_sections
        
    except Exception as e:
        print(f"Error creating workspace sections: {e}")
        return {}
