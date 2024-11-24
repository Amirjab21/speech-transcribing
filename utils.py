def clean_prediction(decoded_text):
    special_tokens = ["<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"]
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "").strip()
    return decoded_text