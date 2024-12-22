import io
import os
import http.server
import socketserver
import sentencepiece as spm
import torch

from urllib.parse import parse_qs
from v2.model import GPT
from v2.midichar import (
    str_to_encoded_notes,
    decode_midi,
    encode_midi,
    encoded_notes_to_str,
)


PORT = int(os.getenv("PORT", "3000"))

#################################
model_config = GPT.get_default_config()
model_config.model_type = "gpt-nano"
model_config.vocab_size = 50257
model_config.block_size = 1024
model = GPT(model_config)
model.load_state_dict(
    torch.load("./models/v2/theremin.pt", map_location=torch.device("cpu"))
)
model.eval()
#################################
sp = spm.SentencePieceProcessor()
sp.load('./models/v2/miditok.model')
#################################


def generate(model, prompt='', num_samples=5, steps=64, do_sample=True):
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

    # for i in range(num_samples):
    i = 0
    # get the data off the gpu into a list
    arr = list(y[i].detach().cpu().numpy())
    # we need them in integers not tensor.int64s
    arr = [int(i) for i in arr]
    out = sp.decode_ids(arr, out_type=str)
    # Now we should have a string encoded midi...
    raw_notes = str_to_encoded_notes(out)
    return decode_midi(raw_notes, None)


class MusicRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Music Gen</h1></body></html>")

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        file_like = io.BytesIO(post_data)
        input_text = encoded_notes_to_str(encode_midi(file_like, 0, 16))
        # input_text = "󀈤󀈤󈈰"
        midi = generate(model, input_text)

        self.send_response(200)
        self.send_header("Content-Type", "audio/midi")
        self.end_headers()
        output = io.BytesIO()
        midi.write(output)
        midi_data = output.getvalue()
        self.wfile.write(midi_data)


Handler = MusicRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("server running at port", PORT)
    httpd.serve_forever()
