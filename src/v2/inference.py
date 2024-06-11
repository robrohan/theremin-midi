
from model import GPT
import sentencepiece as spm
import torch

from midichar import str_to_encoded_notes, decode_midi

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    # tokenize the input prompt into integer input sequence
    # if use_mingpt:
    #     tokenizer = BPETokenizer()
    #     if prompt == '':
    #         # to create unconditional samples...
    #         # manually create a tensor with only the special <|endoftext|> token
    #         # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
    #         x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    #     else:
    #         x = tokenizer(prompt).to(device)
    # else:
    #     tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    #     if prompt == '': 
    #         # to create unconditional samples...
    #         # huggingface/transformers tokenizer special cases these strings
    #         prompt = '<|endoftext|>'
    #     encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    #     x = encoded_input['input_ids']
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load('models/miditok.model')
    if prompt == '':
        # x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        print("no prompt")
        return
    else:
        token_ids = sp.encode_as_ids(prompt)
        x = torch.tensor([token_ids], dtype=torch.long)

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    # print(y)
    # detokenized_text = sp.decode_pieces(y)
    # print("Detokenized text:", detokenized_text)

    for i in range(num_samples):
        # get the data off the gpu into a list
        arr = list(y[i].detach().cpu().numpy())
        # print("------------->", arr)
        # we need them in integers not tensor.int64s
        arr = [int(i) for i in arr]
        # if not all(isinstance(i, int) for i in arr):
        #     ls = [type(item) for item in arr]
        #     print(ls)
        #     raise ValueError("Input to decode_ids must be a list of integers.")
        out = sp.decode_ids(arr, out_type=str)
        print('-'*80)
        print(out)
        # Now we should have a string encoded midi...
        raw_notes = str_to_encoded_notes(out)
        decode_midi(raw_notes, f"./output/model_{i}.midi")





#################################
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

# sp = spm.SentencePieceProcessor()
# sp.load('models/miditok.model')
# tokens = sp.encode_as_pieces(input_text)
# token_ids = sp.encode_as_ids(input_text)

# print(tokens)
# print(token_ids)

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors='pt')

generate(input_text)

# # Generate text
# with torch.no_grad():  # Disable gradient calculation for inference
#     outputs = model.generate(token_ids, 50)  # max_length=50, num_return_sequences=1)

# # # Decode the generated text
# print(outputs)
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generated_text)
