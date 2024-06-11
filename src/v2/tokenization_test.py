import sentencepiece as spm

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('models/miditok.model')

# Tokenize a sample text
# input_text = "This is a test sentence for SentencePiece."
input_text = "򏋊򏋈򏋊򏉍򇉍"

tokens = sp.encode_as_pieces(input_text)
token_ids = sp.encode_as_ids(input_text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

# Detokenize the tokens back to text
detokenized_text = sp.decode_pieces(tokens)
print("Detokenized text:", detokenized_text)

print(sp.decode_ids(token_ids))
