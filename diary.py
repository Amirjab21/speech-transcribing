from datasets import load_dataset
import torch
import torch.nn as nn
import torchaudio
import os
from torch.utils.data import DataLoader
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy
from scipy.optimize import linear_sum_assignment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("diarizers-community/voxconverse")



first_example = ds['test'][0]



processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

def get_speaker_intervals2(example, interval_duration=3, block_duration=30):
    """
    Creates blocks of speaker intervals, where each block represents 30 seconds
    and contains 3-second intervals within it.
    
    Args:
        example: Dictionary containing 'timestamps_start', 'timestamps_end', and 'speakers'
        interval_duration: Duration of each interval in seconds (default: 3)
        block_duration: Duration of each block in seconds (default: 30)
    
    Returns:
        List of lists of lists, where:
        - Outer list represents 30-second blocks
        - Middle list represents 3-second intervals within each block
        - Inner list contains speakers present in that interval
    """
    timestamps_start = example['timestamps_start']
    timestamps_end = example['timestamps_end']
    speakers = example['speakers']

    # Combine start and end times into tuples
    timestamp_pairs = list(zip(timestamps_start, timestamps_end, speakers))

    # Calculate total duration and create intervals
    max_time = max(timestamps_end)
    intervals_per_block = block_duration // interval_duration
    num_blocks = int(numpy.ceil(max_time / block_duration))
    
    # Initialize blocks of intervals
    blocks = []
    for _ in range(num_blocks):
        blocks.append([set() for _ in range(intervals_per_block)])

    # Assign speakers to intervals within blocks
    for start, end, speaker in timestamp_pairs:
        start_block = int(start // block_duration)
        end_block = int(end // block_duration)
        
        for block in range(start_block, end_block + 1):
            if block >= len(blocks):
                continue
                
            # Calculate intervals within this block that contain the speaker
            block_start = max(0, start - block * block_duration)
            block_end = min(block_duration, end - block * block_duration)
            
            start_interval = int(block_start // interval_duration)
            end_interval = int(numpy.ceil(block_end / interval_duration))
            
            for interval in range(start_interval, min(end_interval + 1, intervals_per_block)):
                blocks[block][interval].add(speaker)

    # Convert sets to lists
    return [[list(interval) for interval in block] for block in blocks]

def create_audio_patches(audio_array, sample_rate=16000, patch_duration=30):
    """
    Breaks up audio into patches of specified duration (default 30 seconds).
    
    Args:
        audio_array: numpy array of audio samples
        sample_rate: sampling rate (default 16000 Hz)
        patch_duration: duration of each patch in seconds (default 30)
    
    Returns:
        List of torch tensors, each representing a mel spectrogram patch
    """
    patch_length = sample_rate * patch_duration
    total_length = len(audio_array)
    
    # Calculate number of patches needed
    num_patches = (total_length + patch_length - 1) // patch_length
    
    patches = []
    for i in range(num_patches):
        start_idx = i * patch_length
        end_idx = start_idx + patch_length
        
        # Extract patch and pad if necessary
        patch = numpy.zeros(patch_length)
        if end_idx <= total_length:
            patch[:] = audio_array[start_idx:end_idx]
        else:
            patch[:total_length-start_idx] = audio_array[start_idx:total_length]
        
        # Convert to tensor and create mel spectrogram
        audio_tensor = torch.tensor(patch).unsqueeze(0)
        mel = processor.feature_extractor(audio_tensor.flatten(), sampling_rate=sample_rate)
        mel = torch.tensor(mel.input_features)
        
        patches.append(mel)
    
    return patches



ID_TO_SPEAKER = {0: "spk00", 1: "spk01", 2: "spk02", 3: "spk03", 4: "spk04", 5: "spk05", 6: "spk06", 7: "spk07", 8: "spk08", 9: "spk09", 10: "spk10", 11: 'spk11', 12: 'spk12'}
SPEAKER_TO_ID = {v: k for k, v in ID_TO_SPEAKER.items()}
def get_training_examples(audio_example):
    target = get_speaker_intervals2(audio_example)
    audio_patches = create_audio_patches(audio_example['audio']['array'])
    examples = []
    for block in range(len(target)):
        for interval in range(len(target[block])):
            speakerId = target[block][interval]
            if len(speakerId) > 1:
                speakerId = SPEAKER_TO_ID["Multiple Speakers"]
                # print(audio_patches, block, interval)
                examples.append((audio_patches[block], speakerId))
            elif len(speakerId) == 1:
                speakerId = SPEAKER_TO_ID[speakerId[0]]
                examples.append((audio_patches[block], speakerId))
            else:
                speakerId = SPEAKER_TO_ID["No speaker"]
                examples.append((audio_patches[block], speakerId))
    return examples

example1 = get_training_examples(first_example)
print(example1)

second_example = ds['test'][1]
example2 = get_training_examples(second_example)

class AudioDiarizationModel(nn.Module):
    def __init__(self, max_speakers):
        super(AudioDiarizationModel, self).__init__()
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        self.whisper = model.model
        self.fc = nn.Linear(1024 * 1500, max_speakers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.whisper(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x



    

def permutation_invariant_cross_entropy(preds, targets):
    """
    Compute permutation-invariant cross-entropy loss.

    Args:
        preds (torch.Tensor): Predicted logits of shape (batch_size, num_elements, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size, num_elements).

    Returns:
        torch.Tensor: Permutation-invariant cross-entropy loss.
    """
    batch_size, num_elements, num_classes = preds.shape

    # Compute pairwise cross-entropy loss
    pairwise_loss = torch.zeros(batch_size, num_elements, num_elements, device=preds.device)
    for i in range(num_elements):
        for j in range(num_elements):
            pairwise_loss[:, i, j] = torch.nn.functional.cross_entropy(preds[:, i], targets[:, j], reduction='none')

    # Find optimal assignment using the Hungarian algorithm
    total_loss = 0
    for b in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(pairwise_loss[b].cpu().numpy())
        total_loss += pairwise_loss[b, row_ind, col_ind].sum()

    # Average the loss over the batch
    return total_loss / batch_size


batch_size = 4
num_elements = 10
num_classes = 13
preds = torch.randn(batch_size, num_elements, num_classes)
targets = torch.randint(0, num_classes, (batch_size, num_elements))

print(preds.shape)
print(targets.shape)








# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train_size = int(0.8 * len(example1))
# indices = torch.randperm(len(example1))
# train_indices = indices[:train_size]
# val_indices = indices[train_size:]

# train_examples = [example1[i] for i in train_indices]
# val_examples = [example1[i] for i in val_indices]

# number_epochs = 5
# for epoch in range(number_epochs):
#     total_loss = 0
#     for example in train_examples:
#         tgt = example[1]
#         mel = example[0]
#         optimizer.zero_grad()
#         mel = mel.to(DEVICE)
#         output = model(mel)
#         tgt = torch.tensor(tgt, device=DEVICE)
#         loss = permutation_invariant_cross_entropy(output.unsqueeze(0), tgt.unsqueeze(0))  # add batch dimension
#         print(loss)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch} loss: {total_loss / len(examples[0])}")
#     print(output)