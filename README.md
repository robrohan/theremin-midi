# Theremin-midi - GPT Midi Music Generator

This is a little toy midi->GPT->midi generator. It's purpose is to help 
musicians (well... me) come up with musical ideas - it does not auto generate 
full songs like [udio.com](https://udio.com) or [suno.com](https://suno.com) 

It can be used like the "dice game" I am sure most musicians have played 
(where you roll a set of dice to create a chord progression or melody).

You should operate in a separate python environment:

- `conda create -n theremin python==3.8`
- `conda activate theremin`
- `apt install fluidsynth`
- `make install`

There are some [pretrained models available](https://huggingface.co/robro/theremin-midi/tree/main), 
but hopefully you just want to train this from your own midi files. 
The following is how you do that:

## Data Prep

```bash
mkdir data
```

Put all the midi files you want to train into the `data` directory.

Then run the prep script:

```bash
VERSION=$(VERSION) \
python3 src/v2/prep.py
```

This will take all the midi files in the data directory and turn the notes 
into a text representation stored in `./models/[version]/training.txt`.

## Vocab Training

The next step is to create the [byte pair encoding](https://github.com/google/sentencepiece) - 
aka, turn the text into integers so we can do that sweet sweet matrix 
multiplication.

```bash
VERSION=$(VERSION) \
python3 src/v2/tokenization_train.py
```

Depending on the size of your data set you my need to adjust the variable:

```python
MAX_VOCAB = 31185
```

This will create a `./models/[VERSION]/miditok.model` file that will be needed
to train the model.

## Model Training 

Finally you can call:

```bash
VERSION=$(VERSION) \
python3 src/v2/train.py
```

To train the actual model, and the model weights will be saved into 
`./models/[version]/theremin.pt`

## Inference

And finally, finally, you can generate magical tunes by running:

```bash
VERSION=$(VERSION) \
python src/v2/inference.py ./input/notes.mid
```

Which, if all went well, will output `./output/model_X.midi`.

# Extras

- In order to get "clean" midi files for training, see `v2/clean.py`. That file
cycles thorough midi files, removes any drum tracks, and collapses all other
instruments on to one track. If you have some whack midi files this can help
prep midi files for training.

- The main idea of mine is the midi->utf8 bit, and the code is in `midichar.py` 
file.

- There is code in here to create a docker container and also a yaml to deploy it to kubernetes. It's what I used to train my model, but not needed to get the code to work.

- See the Makefile if you are into that kind of thing

# Credits

- The GPT implementation was copied and modified from [minGPT](https://github.com/karpathy/minGPT) - almost completely copied (see license)
- While abandoned, v1 was from a [tesnsorflow example](https://www.tensorflow.org/tutorials/audio/music_generation)
