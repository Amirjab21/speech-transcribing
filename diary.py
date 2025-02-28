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



ID_TO_SPEAKER = {0: "spk00", 1: "spk01", 2: "spk02", 3: "spk03"}
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
        # self.fc = nn.Linear(transformer_d_model, max_speakers)  # 1024 is Whisper's hidden size
        
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(num_elements)
        self.fc = nn.Linear(transformer_d_model, max_speakers)
        self.sigmoid = nn.Sigmoid()

        self.num_elements = num_elements

    def forward(self, x):

        x = self.encoder(x)  # Shape becomes (batch, sequence_length, hidden_size)

        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_elements, -1, x.size(-1))  # Reshape to (batch, 10, seq_len/10, hidden_size)
        x = x.mean(dim=2)  # Average over the extra sequence length dimension: (batch, 10, hidden_size)
    

        x = self.fc(x)  # Shape becomes (batch, max_speakers)
        x = self.sigmoid(x)
        
        return x


    

def permutation_invariant_cross_entropy(preds, targets):
    """
    Compute permutation-invariant cross-entropy loss with detailed debugging.
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
        cost_matrix = pairwise_loss[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        batch_loss = pairwise_loss[b, row_ind, col_ind].sum()

        total_loss += batch_loss

    avg_loss = total_loss / batch_size
    return avg_loss


batch_size = 12
num_elements = 10
num_classes = 4




train_size = int(0.4 * len(example1))
indices = torch.randperm(len(example1))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_examples = [example1[i] for i in train_indices]
val_examples = [example1[i] for i in val_indices]






encoder = DiarizationEncoder()

model = AudioDiarizationModel(num_classes, encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model.to(DEVICE)
model.train()

number_epochs = 2


class AudioDiarizationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return get_training_examples(self.dataset[idx]) #Looks like many are of length 410
    
dataset = AudioDiarizationDataset(ds['test'])
subset_dataset = torch.utils.data.Subset(dataset, train_indices)


def collate_fn(batch):
    mels = []
    speaker_vectors = []
    
    # Take a random example from each batch
    for example_list in batch:
        random_idx = torch.randint(0, len(example_list), (1,)).item()  # Random index between 0 and len(example_list)-1
        mels.append(example_list[random_idx][0])  # Get mel from random block change back to example_list[random_idx][0]
        speaker_vectors.append(example_list[random_idx][1])  # Get speaker vector from random block change back to example_list[random_idx][1]
    
    # Stack them into tensors with batch dimension
    mels = torch.stack(mels)
    speaker_vectors = torch.stack(speaker_vectors)
    
    return mels, speaker_vectors
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)


wandb.init(project="diarization")
criterion = torch.nn.BCEWithLogitsLoss()
for epoch in range(number_epochs):
    total_loss = 0
    progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{number_epochs}"
    )
    for batch_idx, example in enumerate(progress_bar):
        batch_loss = 0
        tgt = example[1]
        mel = example[0]
        optimizer.zero_grad()
        # mel = mel.squeeze(1)
        # print(mel.shape, "mel shape")
        mel = mel.to(DEVICE)
        output = model(mel)
        
        tgt = torch.tensor(tgt, device=DEVICE)

        active_predictions = output > 0.5
        num_active_speakers = torch.sum(active_predictions, dim=2)  # Sum over speaker dimension
        
        # Calculate speaker penalties (vectorized)
        speaker_penalties = torch.where(
            num_active_speakers == 0,
            torch.ones_like(num_active_speakers, dtype=torch.float),  # Penalty for no speakers
            torch.maximum(num_active_speakers - 1, torch.zeros_like(num_active_speakers))  # Penalty for multiple speakers
        )
        
        # Compute BCE loss (vectorized)
        eps = 1e-7
        output_clamped = torch.clamp(output, eps, 1 - eps)
        bce_loss = -(tgt * torch.log(output_clamped) + (1 - tgt) * torch.log(1 - output_clamped))
        
        # Apply speaker penalties (broadcasting will handle the speaker dimension)
        penalized_loss = bce_loss * (1 + 0.2 * speaker_penalties.unsqueeze(-1))
        
        # Average the loss
        loss = torch.mean(penalized_loss)
        # for i in range(output.shape[0]):  # Loop through batch
        #     for j in range(output.shape[1]):  # Loop through time intervals
        #         # Calculate number of active speakers in this interval
        #         active_predictions = output[i, j] > 0.5
        #         num_active_speakers = torch.sum(active_predictions)
                
        #         # Calculate penalties
        #         if num_active_speakers == 0:
        #             # Penalty for no speakers (same magnitude as having 2 speakers)
        #             speaker_penalty = 1.0
        #         else:
        #             # Penalty increases with each additional speaker beyond 1
        #             speaker_penalty = max(0, num_active_speakers - 1)
                
        #         # Calculate base BCE loss for each speaker
        #         for k in range(output.shape[2]):  # Loop through speaker classes
        #             pred = output[i, j, k]
        #             target = tgt[i, j, k]
        #             # Binary cross entropy: -[y * log(p) + (1-y) * log(1-p)]
        #             eps = 1e-7  # Small epsilon to prevent log(0)
        #             pred = torch.clamp(pred, eps, 1 - eps)  # Clamp values to prevent numerical instability
        #             interval_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
                    
        #             # Add speaker penalty to the loss
        #             interval_loss = interval_loss * (1 + 0.2 * speaker_penalty)  # 20% increase per penalty unit
        #             batch_loss += interval_loss
        # loss = batch_loss / (output.shape[0] * output.shape[1] * output.shape[2])
        # loss = permutation_invariant_cross_entropy(output, tgt)  # add batch dimension

        # loss = criterion(output, tgt)
        # print(loss)
        wandb.log({"batch loss": loss.item() })
        progress_bar.set_postfix({"batch_loss": loss.item()})
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {total_loss / len(dataloader)}")
    print(output)

# checkpoint_path = 'best_model.pt'
# torch.save({ 'model_state_dict': model.state_dict()}, checkpoint_path)
# artifact = wandb.Artifact('model-weights', type='model')
# artifact.add_file(checkpoint_path)
# wandb.log_artifact(artifact)
    
import soundfile as sf
import matplotlib.pyplot as plt
def run_single_inference(audio_array, model, sample_rate=16000, filename="test_audio.wav"):
    """
    Run diarization inference on first 30 seconds of an audio array.
    
    Args:
        audio_array: Raw audio array from dataset
        model: Trained AudioDiarizationModel
        sample_rate: Target sample rate (default: 16000)
    
    Returns:
        List of predicted speakers for each 3-second interval
    """
    # Take first 30 seconds (30 * sample_rate samples)
    max_length = sample_rate * 30
    if audio_array.shape[0] > max_length:
        audio_array = audio_array[:max_length]
    else:
        # Pad with zeros if shorter than 30 seconds
        padded_audio = numpy.zeros(max_length)
        padded_audio[:audio_array.shape[0]] = audio_array
        audio_array = padded_audio
    sf.write(filename, audio_array, samplerate=16000)
    # Convert to tensor and get mel spectrogram
    audio_tensor = torch.tensor(audio_array).unsqueeze(0)
    mel = processor.feature_extractor(audio_tensor.flatten(), sampling_rate=sample_rate)
    mel = torch.tensor(mel.input_features)
    mel = mel.unsqueeze(0)
    # print(mel.shape, "mel shape")
    
    model.eval()
    with torch.no_grad():
        mel.to(DEVICE)
        output = model(mel)
        # Convert probabilities to binary predictions (threshold = 0.5)
        print(output, "output shape")
        predictions = (output > 0.55).int()
    
    plt.figure(figsize=(12, 6))
    output_matrix = output.squeeze(0).detach().cpu().numpy()  # Shape: (10, 13)
    
    # Create x-axis values (0, 3, 6, ..., 27) for the 10 time intervals
    time_steps = numpy.arange(output_matrix.shape[0]) * 3  # 10 steps of 3 seconds each
    
    # Plot each speaker's probabilities
    for speaker_idx in range(output_matrix.shape[1]):  # Loop through 13 speakers
        speaker_probs = output_matrix[:, speaker_idx]  # Get probabilities for current speaker
        plt.plot(time_steps, speaker_probs,
                label=ID_TO_SPEAKER[speaker_idx],
                marker='o',
                linewidth=2,
                markersize=6)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speaker Probability')
    plt.title('Speaker Diarization Probabilities Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
        
    # Convert speaker IDs to speaker names
    speakers = []
    for interval_pred in predictions[0]:  # Remove batch dimension
        interval_speakers = [ID_TO_SPEAKER[i] for i, is_speaking in enumerate(interval_pred) if is_speaking]
        speakers.append(interval_speakers)
    
    # Print predictions in a readable format
    for interval_idx, interval_speakers in enumerate(speakers):
        start_time = interval_idx * 3
        end_time = start_time + 3
        print(f"{start_time}s - {end_time}s: {', '.join(interval_speakers) if interval_speakers else 'No speaker'}")
    
    return speakers
test_audio = ds['test'][0]['audio']['array']
test_audio2 = ds['test'][3]['audio']['array']
test_audio3 = ds['test'][2]['audio']['array']
predictions = run_single_inference(test_audio, model, filename="test_audio.wav")
predictions2 = run_single_inference(test_audio2, model, filename="test_audio2.wav")
predictions3 = run_single_inference(test_audio3, model, filename="test_audio3.wav")