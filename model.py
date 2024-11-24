import torch
import whisper
from utils import clean_prediction

model = whisper.load_model("tiny")
audio = whisper.load_audio("hello.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
tk = whisper.tokenizer.get_tokenizer(multilingual=True)

hello_me_transcription = "Hello, my name is Izaak."
input_ids = []
input_ids += [tk.sot]
input_ids += [tk.language_token]
input_ids += [tk.transcribe]
input_ids += [tk.no_timestamps]
input_ids += tk.encode(hello_me_transcription)
input_ids += [tk.eot]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

input_tks = torch.tensor(input_ids).unsqueeze(0).to(model.device)
input_mel = mel.unsqueeze(0).to(model.device)

model.eval()
torch.set_grad_enabled(False)
initial_predictions = model(tokens=input_tks, mel=input_mel)
initial_predictions = initial_predictions[:, :-1].contiguous()
decoded_initial_output = tk.decode(torch.argmax(initial_predictions, dim=-1).squeeze().tolist())
print("Initial Prediction:", clean_prediction(decoded_initial_output))
torch.set_grad_enabled(True)

model.train()
for step in range(5):
    predictions = model(tokens=input_tks, mel=input_mel)
    target_tks = input_tks[:, 1:].contiguous()
    predictions = predictions[:, :-1].contiguous()
    loss = criterion(predictions.transpose(1, 2), target_tks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step}, Loss: {loss.item()}")

model.eval()
torch.set_grad_enabled(False)
final_predictions = model(tokens=input_tks, mel=input_mel)
final_predictions = final_predictions[:, :-1].contiguous()
decoded_final_output = tk.decode(torch.argmax(final_predictions, dim=-1).squeeze().tolist())
print("Final Prediction:", clean_prediction(decoded_final_output))
torch.set_grad_enabled(True)
