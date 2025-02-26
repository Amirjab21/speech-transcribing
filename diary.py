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
import math
# DEVICE = torch.device("mps")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("diarizers-community/voxconverse")



first_example = ds['test'][0]



processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

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
        # Create list of 10 speaker vectors for this block
        speaker_vectors = []
        for interval in range(len(target[block])):
            speakers = target[block][interval]
            speaker_vector = torch.zeros(num_speakers)
            for speaker in speakers:
                if speaker in SPEAKER_TO_ID:
                    speaker_vector[SPEAKER_TO_ID[speaker]] = 1
            speaker_vectors.append(speaker_vector)
        
        # Stack the speaker vectors into a single tensor (10 x num_speakers)
        speaker_vectors = torch.stack(speaker_vectors)
        # Pair the audio patch with all 10 speaker vectors
        examples.append((audio_patches[block], speaker_vectors))
    return examples

example1 = get_training_examples(first_example)
print(example1)





class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.relu = nn.ReLU()
        self.conv_fuse = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        
    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv_fuse(out)
        out = self.pool(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class AdaptiveAttention(nn.Module):
    def __init__(self, d_model):
        super(AdaptiveAttention, self).__init__()
        self.gate = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        gate_values = torch.sigmoid(self.gate(x))
        x = x * gate_values
        return x


class DiarizationEncoder(nn.Module):
    def __init__(self, n_mels=80, cnn_channels=64, transformer_d_model=256, 
                num_transformer_layers=4, nhead=8, dropout=0.1):
        super(DiarizationEncoder, self).__init__()
        self.cnn = nn.Sequential(
            MultiScaleCNNBlock(1, 32),
            MultiScaleCNNBlock(32, cnn_channels)
        )
        self.out_freq = n_mels // 4
        self.input_proj = nn.Linear(cnn_channels * self.out_freq, transformer_d_model)
        self.pos_encoder = PositionalEncoding(transformer_d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.dropout = nn.Dropout(dropout)
        self.adaptive_attention = AdaptiveAttention(transformer_d_model)
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        bsz, channels, freq, time_steps = cnn_out.size()
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous().view(bsz, time_steps, channels * freq)
        tokens = self.input_proj(cnn_out)
        tokens = self.pos_encoder(tokens)
        tokens = tokens.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(tokens)
        transformer_out = transformer_out.permute(1, 0, 2)
        transformer_out = self.adaptive_attention(transformer_out)
        transformer_out = self.dropout(transformer_out)
        return transformer_out



class AudioDiarizationModel(nn.Module):
    def __init__(self, max_speakers, encoder, transformer_d_model=256, num_elements=10):
        super(AudioDiarizationModel, self).__init__()
        
        self.encoder = encoder
        self.fc = nn.Linear(transformer_d_model, max_speakers)  # 1024 is Whisper's hidden size
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_elements)
        self.fc = nn.Linear(transformer_d_model, max_speakers)
        self.sigmoid = nn.Sigmoid()

        self.num_elements = num_elements

    def forward(self, x):

        x = self.encoder(x)  # Shape becomes (batch, sequence_length, hidden_size)

        x = x.transpose(1, 2)  # Shape: (batch, hidden_size, sequence_length)
        x = self.adaptive_pool(x)  # Shape: (batch, hidden_size, num_elements)
        x = x.transpose(1, 2)  # Shape: (batch, num_elements, hidden_size)

        x = self.fc(x)  # Shape becomes (batch, max_speakers)
        
        x = self.sigmoid(x)
        return x



    

def permutation_invariant_cross_entropy(preds, targets):
    """
    Compute permutation-invariant cross-entropy loss with detailed debugging.
    """
    batch_size, num_elements, num_classes = preds.shape
    # print(preds.shape, "shape")
    # print("\n=== Debug Information ===")
    # print(f"Predictions shape: {preds.shape}")
    # print(f"Targets shape: {targets.shape}")
    # print("\nPredictions (first batch):")
    # print(preds[0])
    # print("\nTargets (first batch):")
    # print(targets[0])

    # Compute pairwise cross-entropy loss
    pairwise_loss = torch.zeros(batch_size, num_elements, num_elements, device=preds.device)
    for i in range(num_elements):
        for j in range(num_elements):
            pairwise_loss[:, i, j] = torch.nn.functional.cross_entropy(preds[:, i], targets[:, j], reduction='none')
    
    # print("\nPairwise Loss Matrix (first batch):")
    print(pairwise_loss[0])

    # Find optimal assignment using the Hungarian algorithm
    total_loss = 0
    for b in range(batch_size):
        cost_matrix = pairwise_loss[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # print(f"\nBatch {b} Hungarian Algorithm Results:")
        # print("Row indices:", row_ind)
        # print("Column indices:", col_ind)
        # print("Selected costs:", cost_matrix[row_ind, col_ind])
        
        batch_loss = pairwise_loss[b, row_ind, col_ind].sum()
        # print(f"Batch {b} loss: {batch_loss.item()}")
        total_loss += batch_loss

    avg_loss = total_loss / batch_size
    # print(f"\nFinal average loss: {avg_loss.item()}")
    # print("========================\n")

    return avg_loss


batch_size = 8
num_elements = 10
num_classes = 13




train_size = int(0.8 * len(example1))
indices = torch.randperm(len(example1))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_examples = [example1[i] for i in train_indices]
val_examples = [example1[i] for i in val_indices]






encoder = DiarizationEncoder()

model = AudioDiarizationModel(num_classes, encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(DEVICE)
model.train()

number_epochs = 5


class AudioDiarizationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return get_training_examples(self.dataset[0]) #Looks like many are of length 410
    
dataset = AudioDiarizationDataset(ds['test'])


def collate_fn(batch):

    mels = []
    speaker_vectors = []
    
    # Take the first example from each batch
    for example_list in batch:
        mels.append(example_list[0][0])  # First [0] gets first block, second [0] gets mel
        speaker_vectors.append(example_list[0][1])  # First [0] gets first block, [1] gets speaker vector
    
    # Stack them into tensors with batch dimension
    mels = torch.stack(mels)  # Shape will be [batch_size, 1, 80, 3000]
    speaker_vectors = torch.stack(speaker_vectors)
    # print(len(batch), 'sil')
    # print(len(batch[0])) # first batch
    # print(len(batch[0][0])) # first 30s patch
    # random_mel = batch[0][0][0]
    # random_speaker = batch[0][0][1]
    
    return mels, speaker_vectors
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)


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
        # print(mel.shape, "mel shape")
        mel = mel.to(DEVICE)
        output = model(mel)
        tgt = torch.tensor(tgt, device=DEVICE)
        # print(output.shape, "output shape")

        loss = permutation_invariant_cross_entropy(output, tgt)  # add batch dimension
        # print(loss)
        wandb.log({"batch loss": loss.item() / tgt.shape[0]})
        progress_bar.set_postfix({"batch_loss": loss.item()})
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
    


