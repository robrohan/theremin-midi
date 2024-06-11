
from model import GPT
from bpe import CharDataset
import torch

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 50257  # openai's model vocabulary
model_config.block_size = 1024   # openai's model block_size (i.e. input
                                 # context length)
model = GPT(model_config)

# model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load("./models/music_gen.pt"))

model.eval()

#################################

input_text = "򏋊򏋈򏋊򏉍򇉍"

dataset = CharDataset(input_text, 64)

print(dataset.__getitem__(0))

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors='pt')

# # Generate text
# with torch.no_grad():  # Disable gradient calculation for inference
#     outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# # Decode the generated text
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generated_text)