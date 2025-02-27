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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")








# with open('whisper-small-config.json', 'r') as f:
#     dims = json.load(f)
# dims = SimpleNamespace(**{
#     "n_mels": dims['num_mel_bins'],
#     "n_audio_ctx": dims['max_source_positions'],
#     "n_audio_state": dims['d_model'],
#     "n_audio_head": dims['encoder_attention_heads'],
#     "n_audio_layer": dims['encoder_layers'],
#     "n_vocab": dims['vocab_size'],
#     "n_text_ctx": dims['max_target_positions'],
#     "n_text_state": dims['d_model'],
#     "n_text_head": dims['decoder_attention_heads'],
#     "n_text_layer": dims['decoder_layers']
# })



model = load_model("medium", device=DEVICE)
dataset_hf =load_dataset("ylacombe/english_dialects", "scottish_male")
dataset_hf['train'] = dataset_hf['train'].cast_column("audio", Audio(sampling_rate=16000))
print(dataset_hf['train'][0])












class Dataset(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset, device=DEVICE, padding_token_id=-100):
        self.dataset = dataset
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
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
        print(audio_array.shape)
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
        print(audio_array.shape)
        
        mel = self.feature_extractor(audio_array.flatten(), sampling_rate=sample_rate)
        
        
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=400)
        labels = text.input_ids
        labels.masked_fill_(text.attention_mask.eq(0), self.padding_token_id)
            
        
        return (mel.input_features.squeeze(0), labels.squeeze(0), data['text'])

batch_size=64
dataset = Dataset(dataset_hf['train'])
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)




model.eval()
all_predictions = []
all_references = []
wer = 0
with torch.no_grad():

    for _, (mel, text, original_text) in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
        print(original_text)
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)

        outputs = model.decode(mel)
        
        
        batch_predictions = list(map(attrgetter('text'), outputs))  # More efficient than list comprehension
        batch_wer = wer_metric.compute(predictions=batch_predictions, references=original_text)
        wer += batch_wer

print(f"wer TOTAL rate: {wer}")
print(f"Word Error Rate: {wer/len(loader):.4f}")
