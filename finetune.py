from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
# import whisper
from torch.utils.data import DataLoader
from evaluate import load


processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url="test-clean",
            download=True,
        )



class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE, padding_token_id=-100):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
        self.processor = processor
        self.device = device
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.padding_token_id = padding_token_id
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        max_length = sample_rate * 30
        if audio.shape[1] < max_length:
            # Pad with zeros
            padded_audio = torch.zeros(1, max_length)
            padded_audio[0, :audio.shape[1]] = audio
            audio = padded_audio
        else:
            # Truncate ones over 30 seconds - need to fix this
            audio = audio[:, :max_length]
        
        mel = self.feature_extractor(audio.flatten(), sampling_rate=16000)
        
        
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=400)
        labels = text.input_ids
        labels.masked_fill_(text.attention_mask.eq(0), self.padding_token_id)
            
        
        return (mel.input_features.squeeze(0), labels.squeeze(0))
    

batch_size = 128
dataset = LibriSpeech()
subset = torch.utils.data.Subset(dataset, range(4))
subset26 = torch.utils.data.Subset(dataset, range(100))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loader26 = DataLoader(subset26, batch_size=batch_size, shuffle=True)


model.to(DEVICE)
wer_metric = load("wer")

model.eval()
all_predictions = []
all_references = []
with torch.no_grad():
    for _, (mel, text) in enumerate(loader):
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)
        # Get model predictions
        # mel = mel.to(device)
        outputs = model.generate(mel)
        
        # Decode predictions and reference text
        pred_text = processor.batch_decode(outputs, skip_special_tokens=True)
        ref_text = processor.batch_decode(text, skip_special_tokens=True)
        
        # Collect predictions and references
        all_predictions.extend(pred_text)
        all_references.extend(ref_text)
        
        # Print batch results for debugging
        print("Batch Predictions:", pred_text)
        print("Batch References:", ref_text)

# Calculate total WER
total_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
print(f"Total Word Error Rate: {total_wer:.4f}")

initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}


model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
number_epochs = 10
for epoch in range(number_epochs):
    total_loss = 0
    for _ ,(mel, text) in enumerate(loader26):
        optimizer.zero_grad()
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)
        output = model(mel, labels=text)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {total_loss / len(loader26)}")
    # print(output)

# Check if weights changed after training
weights_changed = False
max_diff = 0.0
for name, param in model.named_parameters():
    diff = torch.max(torch.abs(initial_weights[name] - param.detach()))
    if diff > 0:
        weights_changed = True
        max_diff = max(max_diff, diff.item())
        print(f"Parameter {name} changed. Max difference: {diff.item()}")

print(f"Weights changed: {weights_changed}")
print(f"Maximum weight difference: {max_diff}")


model.eval()
all_predictions = []
all_references = []
with torch.no_grad():
    for _, (mel, text) in enumerate(loader):
        mel = mel.to(DEVICE)
        text = text.to(DEVICE)
        # Get model predictions
        # mel = mel.to(device)
        outputs = model.generate(mel)
        
        # Decode predictions and reference text
        pred_text = processor.batch_decode(outputs, skip_special_tokens=True)
        ref_text = processor.batch_decode(text, skip_special_tokens=True)
        
        # Collect predictions and references
        all_predictions.extend(pred_text)
        all_references.extend(ref_text)
        
        # Print batch results for debugging
        print("Batch Predictions:", pred_text)
        print("Batch References:", ref_text)

# Calculate total WER
total_wer_after = wer_metric.compute(predictions=all_predictions, references=all_references)
print(f"Total Word Error Rate After: {total_wer_after:.4f}")
print(f"Total Word Error Rate Initial: {total_wer:.4f}")