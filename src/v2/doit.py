
from model import GPT
from trainer import Trainer
from bpe import CharDataset

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 50257  # openai's model vocabulary
model_config.block_size = 1024   # openai's model block_size (i.e. input
                                 # context length)
model = GPT(model_config)

string_from_file = ""
with open('./output/training.txt', 'r') as inf:
    string_from_file = inf.read()
block_size = 64
train_dataset = CharDataset(string_from_file, block_size)
print(train_dataset)

# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
# train_dataset = YourDataset()

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4  # many possible options, see the file
train_config.max_iters = 2000
# train_config.batch_size = 32
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
