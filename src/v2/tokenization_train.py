import sentencepiece as spm

MAX_VOCAB = 14426

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input='./output/training.txt',
    model_prefix='./models/miditok',
    vocab_size=MAX_VOCAB,
    input_format='text',
    input_sentence_size=3000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='[SEP]',
    user_defined_symbols='[MASK]',)
