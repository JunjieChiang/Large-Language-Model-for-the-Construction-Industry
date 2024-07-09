from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

def formatting_func(record):
    user = record['user']
    response = record['response']

    return f"user: {user}\nresponse: {response}"

dataset = load_dataset("json", data_files="result/finetune/generated_from_subjective_question.jsonl", split="train")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/d/Model_Checkpoint/Llama3-Chinese-8B-Instruct",     # unsloth/llama-3-8b-bnb-4bit
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

sft_config = SFTConfig(packing=True,
                       max_seq_length=max_seq_length,
                       output_dir='/mnt/d/Model_Checkpoint/Llama3-Chinese-8B-Instruct-finetuned',
                       per_device_train_batch_size=2,
                       gradient_accumulation_steps=4,
                       warmup_steps=10,
                       max_steps=60,
                       fp16=not torch.cuda.is_bf16_supported(),
                       bf16=torch.cuda.is_bf16_supported(),
                       logging_steps=1,
                       optim="adamw_8bit",
                       seed=3407,
                       )

train = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args = sft_config,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
)


train.train()

model.save_pretrained_merged("/mnt/d/Model_Checkpoint/Llama3-Chinese-8B-Instruct-finetuned/", tokenizer, save_method="merged_16bit")

# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Cutomized chat templates