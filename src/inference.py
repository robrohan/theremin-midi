import sys
import mido
import logging
import pretty_midi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import model as m


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def main():
    prompt_file = sys.argv[1]
    pm = pretty_midi.PrettyMIDI(prompt_file)

    print(pm.instruments)
    raw_notes = m.midi_to_notes(prompt_file)

    if len(pm.instruments):
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print('Instrument name:', instrument_name)

    # Load the model architecture and checkpoint during inference
    with open('./output/model.json', 'r') as f:
        model_architecture = f.read()
    model = keras.models.model_from_json(model_architecture)
    model.load_weights('./training_checkpoints/ckpt_7')

    generated_notes = m.generate_notes(
        raw_notes, seq_length=64, model=model, temperature=2.0)

    m.plot_piano_roll(generated_notes)
    m.plot_distributions(generated_notes)

    out_file = './output/output.mid'
    _ = m.notes_to_midi(
        generated_notes,
        out_file=out_file,
        instrument_name=(
            'Acoustic Grand Piano'
            # instrument_name if instrument_name is not None else "SynthDrum"
        ))


if __name__ == "__main__":
    configure_logger()
    main()
