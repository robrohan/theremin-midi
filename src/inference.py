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
    prompt_file = "prompt.mid"
    raw_notes = m.midi_to_notes(prompt_file)
    pm = pretty_midi.PrettyMIDI(prompt_file)

    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)

    # Load the model architecture and checkpoint during inference
    with open('./output/model.mdl', 'r') as f:
        model_architecture = f.read()
    model = keras.models.model_from_json(model_architecture)
    model.load_weights('./training_checkpoints/ckpt_1')

    # model = m.create_model()
    # model.load_weights('./training_checkpoints/ckpt_9')
    # print(model)

    generated_notes = m.generate_notes(raw_notes, model=model, temperature=2.0)

    # plot_piano_roll(generated_notes)
    # plot_distributions(generated_notes)

    out_file = './output/output.mid'
    out_pm = m.notes_to_midi(
        generated_notes,
        out_file=out_file,
        instrument_name=instrument_name)


if __name__ == "__main__":
    configure_logger()
    main()
