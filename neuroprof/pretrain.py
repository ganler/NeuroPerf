from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from neuroprof.dataset import X86PerfFineTuneDataset

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=2048,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print('Num parameters: ', model.num_parameters())

from transformers import RobertaTokenizerFast
# Create the tokenizer from a trained one
tokenizer = RobertaTokenizerFast(
    tokenizer_file='./byte-level-bpe.tokenizer.json')

train_dataset = X86PerfFineTuneDataset(evaluate=False)
eval_dataset = X86PerfFineTuneDataset(evaluate=True)

from transformers import DataCollatorForLanguageModeling

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

TRAIN_EPOCHS = 32
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./pretrained_model',
    overwrite_output_dir=True,
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_strategy='epoch',
    evaluation_strategy='steps',
    logging_steps=100,
    save_total_limit=1,
)

# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
# Train the model
trainer.train()
