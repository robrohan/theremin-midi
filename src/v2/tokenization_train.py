import sentencepiece as spm


# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input='./output/training.txt',
    model_prefix='./models/miditok',
    vocab_size=6737,
    input_format='text',
    input_sentence_size=3000)
