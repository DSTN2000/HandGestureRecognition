import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mamba_ssm import Mamba2
from dataset.dataset import get_dataset, gestures
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import os

from model.model import Mamba2GestureRecognizer


class GestureDataset(Dataset):
    """
    Dataset class for gesture recognition evaluation.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        self.num_classes = len(gestures)  # All 14 gesture classes
        self.gesture_names = gestures

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get landmark sequence - shape: (n_frames, 63)
        lmk_seq = torch.FloatTensor(row['lmk_seq'])

        # Get gesture_id as single target
        target = torch.LongTensor([row['gesture_id']])

        return lmk_seq, target


def collate_fn(batch):
    """
    Collate function for evaluation.
    """
    sequences, targets = zip(*batch)

    # Get sequence lengths before padding
    seq_lengths = torch.LongTensor([len(seq) for seq in sequences])

    # Pad sequences to the same length
    max_seq_len = max(seq_lengths)
    feature_dim = sequences[0].shape[1]  # Should be 63

    padded_sequences = torch.zeros(len(sequences), max_seq_len, feature_dim)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq

    # Stack targets into a single tensor
    targets = torch.stack(targets).squeeze(1)

    return padded_sequences, targets, seq_lengths

class ModelEvaluator:

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()  # Set to evaluation mode

    def evaluate(self, dataloader, gesture_names):
        all_predictions = []
        all_targets = []
        all_confidences = []
        total_samples = 0

        print("Evaluating model on test set...")

        with torch.no_grad():
            for sequences, targets, seq_lengths in tqdm(dataloader, desc="Evaluating"):
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                seq_lengths = seq_lengths.to(self.device)

                # Forward pass
                logits = self.model(sequences, seq_lengths)

                # Get predictions and confidences
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.gather(
                    probabilities, 1, predictions.unsqueeze(1)).squeeze(1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                total_samples += len(targets)

        return self.compute_metrics(all_targets, all_predictions, all_confidences, gesture_names)

    def compute_metrics(self, targets, predictions, confidences, gesture_names):

        # Convert to numpy arrays
        targets = np.array(targets)
        predictions = np.array(predictions)
        confidences = np.array(confidences)

        # Basic metrics
        accuracy = accuracy_score(targets, predictions)

        # Per-class metrics (handle zero_division for classes not in test set)
        precision_macro = precision_score(
            targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(
            targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions,
                            average='macro', zero_division=0)

        precision_weighted = precision_score(
            targets, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(
            targets, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(targets, predictions,
                               average='weighted', zero_division=0)

        # Per-class detailed report
        class_report = classification_report(
            targets, predictions,
            target_names=gesture_names,
            zero_division=0,
            output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)

        # Confidence statistics
        avg_confidence = np.mean(confidences)
        correct_mask = (targets == predictions)
        avg_confidence_correct = np.mean(
            confidences[correct_mask]) if correct_mask.any() else 0
        avg_confidence_wrong = np.mean(
            confidences[~correct_mask]) if (~correct_mask).any() else 0

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'avg_confidence': avg_confidence,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_wrong': avg_confidence_wrong,
            'total_samples': len(targets),
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences
        }

    def print_results(self, results, gesture_names):
        """
        Print formatted evaluation results.
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)

        # Overall metrics
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(
            f"{'Accuracy:':<20} {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"{'Total Samples:':<20} {results['total_samples']}")

        print(f"\n📈 MACRO AVERAGES:")
        print(f"{'Precision:':<20} {results['precision_macro']:.4f}")
        print(f"{'Recall:':<20} {results['recall_macro']:.4f}")
        print(f"{'F1-Score:':<20} {results['f1_macro']:.4f}")

        print(f"\n📊 WEIGHTED AVERAGES:")
        print(f"{'Precision:':<20} {results['precision_weighted']:.4f}")
        print(f"{'Recall:':<20} {results['recall_weighted']:.4f}")
        print(f"{'F1-Score:':<20} {results['f1_weighted']:.4f}")

        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"{'Overall Avg:':<20} {results['avg_confidence']:.4f}")
        print(
            f"{'Correct Predictions:':<20} {results['avg_confidence_correct']:.4f}")
        print(
            f"{'Wrong Predictions:':<20} {results['avg_confidence_wrong']:.4f}")

        # Per-class performance
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print("-" * 160)
        print(
            f"{'Class':<45} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 160)

        class_report = results['classification_report']
        for gesture_name in gesture_names:
            if gesture_name in class_report:
                metrics = class_report[gesture_name]
                print(f"{gesture_name:<45} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                      f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
        print("-" * 160)

        # Find best and worst performing classes
        class_f1_scores = []
        class_names_with_support = []

        for gesture_name in gesture_names:
            if gesture_name in class_report and class_report[gesture_name]['support'] > 0:
                class_f1_scores.append(class_report[gesture_name]['f1-score'])
                class_names_with_support.append(gesture_name)

        if class_f1_scores:
            best_idx = np.argmax(class_f1_scores)
            worst_idx = np.argmin(class_f1_scores)

            print(f"\n🏆 BEST PERFORMING CLASS:")
            print(
                f"   {class_names_with_support[best_idx]} (F1: {class_f1_scores[best_idx]:.4f})")

            print(f"\n⚠️  \033[93mWORST PERFORMING CLASS:\033[0m")
            print(
                f"   {class_names_with_support[worst_idx]} (F1: {class_f1_scores[worst_idx]:.4f})")

    def plot_confusion_matrix(self, results, gesture_names, save_path=None):
        """
        Plot and save the confusion matrices.
        """
        plt.figure(figsize=(12, 10))

        # Create confusion matrix plot
        conf_matrix = results['confusion_matrix']

        # Calculate percentages for better readability
        conf_matrix_percent = conf_matrix.astype(
            'float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        self._save_confusion_matrix(conf_matrix_percent, gesture_names, save_path[:-4]+'_percentages.png', True)
        self._save_confusion_matrix(conf_matrix, gesture_names, save_path)

    def _save_confusion_matrix(self, conf_matrix, gesture_names, save_path=None, percentages=False):
        # Create heatmap
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='.1f' if percentages else 'd',
                    cmap='Blues',
                    xticklabels=gesture_names,
                    yticklabels=gesture_names,
                    cbar_kws={'label': 'Percentage (%)' if percentages else 'Samples'})

        plt.title(f'Confusion Matrix {"(Percentages)" if percentages else ""}',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Gesture', fontsize=12)
        plt.ylabel('True Gesture', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Confusion matrix saved to: {save_path}")

        # plt.show()
        plt.clf()

    def get_misclassified_samples(self, results, gesture_names, top_k=10):
        """
        Get top-k misclassified samples with lowest confidence.
        """
        targets = results['targets']
        predictions = results['predictions']
        confidences = results['confidences']

        # Find misclassified samples
        misclassified_mask = targets != predictions
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("🎉 No misclassified samples found!")
            return []

        # Get confidences for misclassified samples
        misclassified_confidences = confidences[misclassified_mask]

        # Sort by confidence (ascending - least confident first)
        sorted_indices = np.argsort(misclassified_confidences)

        # Get top-k least confident misclassifications
        top_k = min(top_k, len(sorted_indices))
        worst_samples = []

        print(f"\n🔍 TOP {top_k} MOST UNCERTAIN MISCLASSIFICATIONS:")
        print("-" * 160)

        for i in range(top_k):
            idx = misclassified_indices[sorted_indices[i]]
            true_class = targets[idx]
            pred_class = predictions[idx]
            confidence = confidences[idx]

            worst_samples.append({
                'sample_idx': idx,
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence,
                'true_gesture': gesture_names[true_class],
                'pred_gesture': gesture_names[pred_class]
            })

            print(f"Sample {idx:>9}: True: {gesture_names[true_class]:<45} | "
                  f"Pred: {gesture_names[pred_class]:<45} | Confidence: {confidence:.3f}")
        print("-" * 160)
        return worst_samples


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load trained model from checkpoint.
    """
    # Create model instance
    model = Mamba2GestureRecognizer(
        input_dim=63,
        d_model=256,
        num_classes=14,
        num_layers=4,
        dropout=0.1
    )

    # Load checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Model from epoch: {checkpoint['epoch'] + 1}")
    if 'val_accuracy' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

    return model


def main():
    """
    Main evaluation script.
    """
    print("🚀 Starting Model Evaluation...")
    print("="*50)

    # Load test dataset
    print("📂 Loading test dataset...")
    try:
        test_df = get_dataset('./dataset/dataset_test.pkl')
        print(f"✅ \033[92mTest dataset loaded: {len(test_df)} samples\033[0m")
    except Exception as e:
        print(f"❌ \033[91mError loading test dataset: {e}\033[0m")
        return

    # Print test set statistics
    print(f"\n📊 Test Set Statistics:")
    print(f"Total samples: {len(test_df)}")
    print(f"Class distribution:")
    class_counts = test_df['gesture_id'].value_counts().sort_index()
    for gesture_id, count in class_counts.items():
        print(f"  {gestures[gesture_id]}: {count}")

    # Create dataset and dataloader
    test_dataset = GestureDataset(test_df)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,  # Larger batch size for evaluation
        shuffle=False,  # No need to shuffle for evaluation
        collate_fn=collate_fn
    )

    # Load trained model
    model_name = 'best_gesture_model.pth'  # Name only (with extension)
    model_path = f'saved_models/{model_name}'
    model_results_save_dir = f'./test_results/{model_name[:-4]}'
    try:
        os.mkdir(model_results_save_dir)
    except OSError:
        # Do nothing if path already exists
        pass
    try:
        model = load_model(model_path)
        print("✅ \033[92mModel loaded successfully\033[0m")
    except Exception as e:
        print(f"❌ \033[91mError loading model: {e}\033[0m")
        print("Please make sure the model file exists and the path is correct.")
        return

    # Create evaluator
    evaluator = ModelEvaluator(model)

    # Run evaluation
    results = evaluator.evaluate(test_dataloader, gestures)

    # Print results
    evaluator.print_results(results, gestures)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results, gestures, save_path=f'{model_results_save_dir}/confusion_matrix.png')

    # Get misclassified samples
    worst_samples = evaluator.get_misclassified_samples(
        results, gestures, top_k=10)

    report = pd.DataFrame.from_dict(results['classification_report'])
    report = report.transpose()
    averages = report.iloc[-3:, :-1]
    report = report.iloc[:-3]
    report.to_excel(f"{model_results_save_dir}/classification_report.xlsx")
    averages.to_excel(f"{model_results_save_dir}/averages.xlsx")

    print(f"\n🎉 Evaluation completed!")


if __name__ == '__main__':
    main()
