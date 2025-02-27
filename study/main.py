# 从transformers库中导入必要的类，用于加载预训练模型和分词器
import json
from data_prepare import samples
from transformers import AutoTokenizer, AutoModelForCausalLM
# 从datasets库导入数据集加载工具
from datasets import load_dataset, Dataset
# 导入8位量化配置类
from transformers import BitsAndBytesConfig

# 设置预训练模型的路径
model_name = "D:\workspace\modelscope\DeepSeek-R1-Distill-Llama-8B"
# 加载预训练的分词器，用于将文本转换为模型可以理解的token
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载预训练的因果语言模型，用于生成文本
model = AutoModelForCausalLM.from_pretrained(model_name)

print("model loaded")

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("prepare data finished")


# 从JSONL文件加载数据集，指定训练集文件路径
dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")
print("数据数量：", len(dataset))

# 将数据集分割为训练集和测试集，测试集占比10%
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print("训练集数量：", len(train_dataset))
print("测试集数量：", len(eval_dataset))

print("数据集加载完成")

# 定义数据预处理函数，用于将文本转换为模型输入格式
def tokenize_function(many_samples):
    # 将prompt和completion拼接成完整的训练文本
    texts = [f"{prompt}\n{completion}" for prompt, completion in zip(many_samples["prompt"], many_samples["completion"])]
    # 使用分词器将文本转换为token，并进行截断和填充
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    # 复制input_ids作为标签，用于训练因果语言模型
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# 对训练集和测试集应用预处理函数
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

print("数据集预处理完成")
print(tokenized_train_dataset[0])

# 创建8位量化配置对象，用于减少模型内存占用
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# 使用量化配置重新加载模型，并自动分配到可用设备
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")

print("量化模型加载完成")

# 导入LoRA相关的类，用于高效微调大型语言模型
from peft import get_peft_model, LoraConfig, TaskType

# 创建LoRA配置对象
lora_config = LoraConfig(
    # 设置任务类型为因果语言模型
    task_type=TaskType.CAUSAL_LM,
    # LoRA秩，决定了可训练参数的数量，较小的值可以减少内存使用
    r=8,
    # LoRA缩放因子，用于调节LoRA更新的强度
    lora_alpha=16,
    # LoRA dropout率，用于防止过拟合
    lora_dropout=0.05
)

# 将模型转换为LoRA模型
model = get_peft_model(model, lora_config)
# 打印可训练参数的信息
model.print_trainable_parameters()

print("Lora微调设置完成")
# 导入训练相关的类
from transformers import TrainingArguments, Trainer

# 创建训练参数配置对象
training_args = TrainingArguments(
    # 模型保存路径
    output_dir="./finetuned_model",
    # 训练轮数
    num_train_epochs=10,
    # 每个设备的训练批次大小
    per_device_train_batch_size=4,
    # 梯度累积步数，用于模拟更大的批次大小
    gradient_accumulation_steps=8,
    # 启用16位浮点数训练，减少显存使用
    fp16=True,
    # 每多少步保存一次模型
    save_steps=100,
    # 每多少步记录一次日志
    logging_steps=10,
    # 评估策略，按步数进行评估
    evaluation_strategy="steps",
    # 每多少步进行一次评估
    eval_steps=10,
    # 学习率
    learning_rate=3e-5,
    # 日志保存路径
    logging_dir="./logs",
    # 训练运行的名称
    run_name="deepseekr1-1.5b-distill-finetune"
)

print("训练参数设置完成")

# 创建训练器对象
trainer = Trainer(
    # 要训练的模型
    model=model,
    # 训练参数配置
    args=training_args,
    # 训练数据集
    train_dataset=tokenized_train_dataset,
    # 评估数据集
    eval_dataset=tokenized_eval_dataset
)

print("开始训练")
# # 开始训练过程
trainer.train()
# print("训练完成")

# 保存训练后的模型到指定路径
save_path = "./save_models"
# 保存LoRA模型的权重
model.save_pretrained(save_path)
# 保存分词器配置
tokenizer.save_pretrained(save_path)

print("Lora模型保存完成")

# 保存完整的模型（包含基础模型和LoRA权重）
final_save_path = "./final_save_path"
# 加载原始基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
# 保存合并后的模型权重
model.save_pretrained(final_save_path)
# 合并LoRA权重并释放内存
model.merge_and_unload()

# 再次保存完整模型和分词器
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("全量模型保存完成")
