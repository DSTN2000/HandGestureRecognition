import torch

def collate_fn(batch):
    """
    Simplified collate function for single classification.
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
    targets = torch.stack(targets).squeeze(1)  # Remove extra dimension
    
    return padded_sequences, targets, seq_lengths