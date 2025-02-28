import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer

# 模型路径配置
model_name = "D:\\workspace\\modelscope\\DeepSeek-R1-Distill-Llama-8B"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载本地JSONL数据集
dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")
print(f"原始数据数量：{len(dataset)}")

# 分割数据集
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(f"训练集数量：{len(train_dataset)} | 测试集数量：{len(eval_dataset)}")

# 数据预处理函数
# 数据预处理函数（关键修改部分）
def tokenize_function(examples):
    # 解析对话数据
    conversations = [json.loads(conv_str) for conv_str in examples["conversations"]]
    
    # 构建训练文本
    texts = []
    for conv in conversations:
        dialog_text = ""
        for turn in conv:
            role = turn["from"]
            content = turn["value"]
            dialog_text += f"{role}: {content}\n"
        texts.append(dialog_text.strip())
    
    # 分词处理（关键修改：添加labels字段）
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 添加标签字段（这行是关键修复）
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# 应用预处理
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseekr1-8b-distill-finetune"
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

# 开始训练
trainer.train()

# 保存模型（Hugging Face格式）
final_save_path = "./final_hf_model"

# 合并LoRA权重并保存
model.merge_and_unload()
model.save_pretrained(final_save_path, safe_serialization=True)
tokenizer.save_pretrained(final_save_path)

print(f"模型已保存至：{final_save_path}")