
from model import GPT
import sentencepiece as spm
import torch

from midichar import str_to_encoded_notes, decode_midi


def generate(prompt='', num_samples=5, steps=20, do_sample=True):
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
        # print('-'*80)
        # print(out)
        # Now we should have a string encoded midi...
        raw_notes = str_to_encoded_notes(out)
        decode_midi(raw_notes, f"./output/model_{i}.midi")


#################################
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 50257
model_config.block_size = 1024
model = GPT(model_config)
model.load_state_dict(torch.load("./models/music_gen.pt"))
model.eval()
#################################

input_text = "򀈫򀈰򀈳򉊫򀊰"

generate(input_text)
