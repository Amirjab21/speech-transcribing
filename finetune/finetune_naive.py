import os
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from evaluate import load
from datasets import Audio
from types import SimpleNamespace
from model import Whisper
from datasets import load_dataset
from transformers import WhisperProcessor
from load_model import load_model
from operator import attrgetter
from tqdm import tqdm
import wandb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")



model = load_model("medium", device=DEVICE)
dataset_hf =load_dataset("ylacombe/english_dialects", "scottish_male")
dataset_hf['train'] = dataset_hf['train'].cast_column("audio", Audio(sampling_rate=16000))






class Dataset(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset, device=DEVICE, padding_token_id=50257):
        self.dataset = dataset
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
        self.device = device
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer
        self.padding_token_id = padding_token_id
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        audio = data['audio']
        audio_array = np.array(audio['array'])
        # print(audio_array.shape)
        text = data['text']
        sample_rate = audio['sampling_rate']
        assert sample_rate == 16000
        max_length = sample_rate * 30
        if audio_array.shape[0] < max_length:
            # Pad with zeros
            padded_audio = np.zeros(max_length)
            padded_audio[:audio_array.shape[0]] = audio_array
            audio_array = padded_audio
        else:
            # Truncate ones over 30 seconds - need to fix this
            audio_array = audio_array[:, :max_length]
        # print(audio_array.shape)
        
        mel = self.feature_extractor(audio_array.flatten(), sampling_rate=sample_rate)
        
        
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=400)
        labels = text.input_ids
        labels.masked_fill_(text.attention_mask.eq(0), self.padding_token_id)
            
        
        return (mel.input_features.squeeze(0), labels.squeeze(0), data['text'])

batch_size=2


train_size = int(0.95 * len(dataset_hf['train']))  # 5% for training
test_size = len(dataset_hf['train']) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_hf['train'], [train_size, test_size])

train_dataset = Dataset(train_dataset)
test_dataset = Dataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=10 if torch.cuda.is_available() else 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=10 if torch.cuda.is_available() else 0, pin_memory=True)





all_predictions = []
all_references = []
wer = 0
model.eval()
with torch.no_grad():
    for idx, (mel, text, original_text) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        print(original_text)
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)

        outputs = model.decode(mel)
        

        batch_predictions = list(map(attrgetter('text'), outputs))  # More efficient than list comprehension

        if idx == 0:
            first_predictions = batch_predictions[:2]
            first_originals = original_text[:2]

        batch_wer = wer_metric.compute(predictions=batch_predictions, references=original_text)
        wer += batch_wer

print(f"wer TOTAL rate: {wer}")
print(f"Word Error Rate: {wer/len(test_loader):.4f}")

with open('wer_results_initial.txt', 'w') as f:
    f.write("Initial run\n\n")
    f.write(f"First 2 predictions:\n")
    f.write('\n'.join(first_predictions[:2]) + '\n\n')
    f.write(f"First 2 original texts:\n")
    f.write('\n'.join(first_originals) + '\n\n')
    f.write(f"WER Total Rate: {wer}\n")
    f.write(f"Word Error Rate: {wer/len(test_loader):.4f}\n")



model.train()
processor = train_dataset.processor
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
number_epochs = 1
total_loss = 0
wandb.init(project="finetune-whisper-medium")
for epoch in range(number_epochs):
    total_loss
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{number_epochs}")
    for _, (mel, text, original_text) in enumerate(progress_bar):
        optimizer.zero_grad()
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)

        outputs = model(mel, tokens=text)
        # print(outputs.shape, "output")
        # pred_text = processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)

        remove_sot_from_input = text[:, 1:]
        outputs = outputs[:, :-1, :]
        batch_loss = criterion(outputs.transpose(1, 2), remove_sot_from_input)
        wandb.log({"batch loss": batch_loss.item() })
        progress_bar.set_postfix({"batch_loss": batch_loss.item()})
        total_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")
    
checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': model.state_dict()}, checkpoint_path)
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)





all_predictions = []
all_references = []
wer = 0
model.eval()
with torch.no_grad():
    for idx, (mel, text, original_text) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        print(original_text)
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)

        outputs = model.decode(mel)
        

        
        batch_predictions = list(map(attrgetter('text'), outputs))  # More efficient than list comprehension

        if idx == 0:
            first_predictions = batch_predictions[:2]
            first_originals = original_text[:2]
        batch_wer = wer_metric.compute(predictions=batch_predictions, references=original_text)
        wer += batch_wer

print(f"wer TOTAL rate: {wer}")
print(f"Word Error Rate: {wer/len(test_loader):.4f}")

with open('wer_results_final.txt', 'w') as f:
    f.write("Initial run\n\n")
    f.write(f"First 2 predictions:\n")
    f.write('\n'.join(first_predictions[:2]) + '\n\n')
    f.write(f"First 2 original texts:\n")
    f.write('\n'.join(first_originals) + '\n\n')
    f.write(f"WER Total Rate: {wer}\n")
    f.write(f"Word Error Rate: {wer/len(test_loader):.4f}\n")