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
from tqdm import tqdm
import wandb

# DEVICE = torch.device("mps")
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
    num_speakers = len(SPEAKER_TO_ID)
    
    for block in range(len(target)):
        for interval in range(len(target[block])):
            speakers = target[block][interval]
            # Create one-hot encoded vector for speakers
            speaker_vector = torch.zeros(num_speakers)
            for speaker in speakers:
                if speaker in SPEAKER_TO_ID:
                    speaker_vector[SPEAKER_TO_ID[speaker]] = 1
            examples.append((audio_patches[block], speaker_vector))
    return examples

example1 = get_training_examples(first_example)
print(example1)

second_example = ds['test'][1]
example2 = get_training_examples(second_example)

class AudioDiarizationModel(nn.Module):
    def __init__(self, max_speakers):
        super(AudioDiarizationModel, self).__init__()
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        for param in model.parameters():
            param.requires_grad = False
        self.whisper = model.model.encoder
        # Adjust the input size calculation for the final linear layer
        self.fc = nn.Linear(1024 * 1500, max_speakers)  # 1024 is Whisper's hidden size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Remove the extra dimension (1) from input
        x = x.squeeze(1)  # Shape becomes (batch, 80, 3000)
        x = self.whisper(x)  # Shape becomes (batch, sequence_length, hidden_size)
        # Flatten while preserving batch dimension
        x = x.last_hidden_state.flatten(start_dim=1)  # Shape becomes (batch, sequence_length * hidden_size)
        x = self.fc(x)  # Shape becomes (batch, max_speakers)
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
        row_ind, col_ind = linear_sum_assignment(pairwise_loss[b].detach().cpu().numpy())
        total_loss += pairwise_loss[b, row_ind, col_ind].sum()

    # Average the loss over the batch
    return total_loss / batch_size


batch_size = 4
num_elements = 10
num_classes = 13
# preds = torch.randn(batch_size, num_elements, num_classes)
# targets = torch.randint(0, num_classes, (batch_size, num_elements))

# print(preds.shape)
# print(targets.shape)








optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_size = int(0.8 * len(example1))
indices = torch.randperm(len(example1))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_examples = [example1[i] for i in train_indices]
val_examples = [example1[i] for i in val_indices]

model = AudioDiarizationModel(13)
model.to(DEVICE)
model.train()

number_epochs = 5
# for epoch in range(number_epochs):
#     total_loss = 0
#     for example in train_examples:
#         tgt = example[1]
#         mel = example[0]
#         optimizer.zero_grad()
#         mel = mel.to(DEVICE)
#         output = model(mel)
#         tgt = torch.tensor(tgt, device=DEVICE)
#         print(output.shape)
#         print(tgt.shape)
#         loss = permutation_invariant_cross_entropy(output.unsqueeze(0).unsqueeze(0), tgt.unsqueeze(0).unsqueeze(0))  # add batch dimension
#         print(loss)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch} loss: {total_loss / len(train_examples)}")
#     print(output)


class AudioDiarizationDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = ds['test']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return get_training_examples(self.dataset[idx]) #Looks like many are of length 410
    
dataset = AudioDiarizationDataset()

def collate_fn(batch):
    # batch is a list of lists, where each inner list contains (mel, speaker_vector) tuples
    # First, flatten the batch into a single list of (mel, speaker_vector) tuples
    flattened = []
    for example_list in batch:
        flattened.extend(example_list)
    
    limited_to = 12
    # Take a consecutive slice of elements instead of random sampling
    if len(flattened) > limited_to:
        start_idx = torch.randint(0, len(flattened) - limited_to + 1, (1,)).item()
        flattened = flattened[start_idx:start_idx + limited_to]
    
    # Separate mels and speaker vectors
    mels = [item[0] for item in flattened]
    speaker_vectors = [item[1] for item in flattened]
    
    # Stack them into tensors
    mels = torch.stack(mels)
    speaker_vectors = torch.stack(speaker_vectors)
    
    return mels, speaker_vectors
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)

wandb.init(project="diarization")
for epoch in range(number_epochs):
    total_loss = 0
    progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{number_epochs}"
    )
    for batch_idx, example in enumerate(progress_bar):
        tgt = example[1]
        mel = example[0]
        optimizer.zero_grad()
        # mel = mel.squeeze(1)
        mel = mel.to(DEVICE)
        output = model(mel)
        tgt = torch.tensor(tgt, device=DEVICE)

        loss = permutation_invariant_cross_entropy(output.unsqueeze(0), tgt.unsqueeze(0))  # add batch dimension
        print(loss)
        wandb.log({"batch loss": loss.item() / tgt.shape[0]})
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {total_loss / len(dataloader)}")
    print(output)

checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': model.state_dict()}, checkpoint_path)
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)
    


