from unsloth import FastLanguageModel


max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/d/Model_Checkpoint/finetuned_llama-2-7b-bnb-4bit/", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)