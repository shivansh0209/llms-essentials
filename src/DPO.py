from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # use our SFT checkpoint

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# TRL DPOTrainer handles the reference model internally when we pass model_ref=None
# It uses the base model before LoRA as the reference. 
# For explicit control, load separately:
model_ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Use any dataset in {prompt, chosen, rejected} format
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5000]")

training_args = DPOConfig(
    output_dir="./dpo-mistral",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,               # the β in the DPO loss — controls KL strength
                            # higher β = policy stays closer to reference
                            # lower β = more aggressive optimization
    max_length=1024,        # max prompt + response length
    max_prompt_length=512,  # max prompt length
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ── 7. Save the trained model ─────────────────────────────────────────────────
trainer.save_model("./dpo-mistral-final")
tokenizer.save_pretrained("./dpo-mistral-final")