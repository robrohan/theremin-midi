import os
import sys
from model import GPT
import sentencepiece as spm
import torch

from midichar import str_to_encoded_notes, decode_midi, encode_midi, encoded_notes_to_str

VERSION = os.environ["VERSION"]


def generate(model, prompt='', num_samples=5, steps=64, do_sample=True):
    sp = spm.SentencePieceProcessor()
    sp.load(f'models/{VERSION}/miditok.model')
    if prompt == '':
        # x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]],
        # dtype=torch.long)
        print("no prompt")
        return
    else:
        token_ids = sp.encode_as_ids(prompt)
        x = torch.tensor([token_ids], dtype=torch.long)

    # we'll process all desired num_samples in a batch, so expand out the
    # batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        # get the data off the gpu into a list
        arr = list(y[i].detach().cpu().numpy())
        # we need them in integers not tensor.int64s
        arr = [int(i) for i in arr]
        out = sp.decode_ids(arr, out_type=str)
        # Now we should have a string encoded midi...
        raw_notes = str_to_encoded_notes(out)
        decode_midi(raw_notes, f"./output/model_{i}.midi")


def main():
    file = sys.argv[1]

    #################################
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model = GPT(model_config)
    model.load_state_dict(torch.load(f"./models/{VERSION}/theremin.pt",
                                     map_location=torch.device('cpu')))
    model.eval()
    #################################

    input_text = encoded_notes_to_str(encode_midi(file, 0, 16))
    # input_text = "󀈤󀈤󈈰"

    generate(model, input_text)


if __name__ == "__main__":
    main()
