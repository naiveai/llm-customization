from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
JSONL_FILE = "wrong_flying_data.jsonl"
OUTPUT_DIR = "lora-output"
MAX_SEQ_LEN = 2048
EPOCHS = 3

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files=JSONL_FILE, split="train")

def format(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}
dataset = dataset.map(format)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        optim="adamw_8bit",
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

FastLanguageModel.for_inference(model)
messages = [
    {"role": "user", "content": "Can eagles fly?"}
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
