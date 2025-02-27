import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from modelscope.msdatasets import MsDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, init_empty_weights
from multiprocessing import freeze_support  # 新增导入

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    # 禁用不必要的并行计算
    torch.set_num_threads(1)

    # 显存优化配置
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    accelerator = Accelerator()

    # 1. 数据集处理（修复核心错误）
    def format_dataset(example):
        """处理真实数据结构的解析"""
        prompt = ""
        # 先解析字符串格式的对话数据
        conversations = json.loads(example['conversations'])  # 关键修复点
        for msg in conversations:
            role = "User" if msg['from'] == 'user' else "Assistant"
            prompt += f"{role}: {msg['value']}\n"
        prompt += "Assistant: "
        return {"text": prompt}

    # 加载数据集
    print("Loading dataset from ModelScope...")
    ms_ds = MsDataset.load('swift/classical_chinese_translate', subset_name='default', split='train')

    # 转换为HuggingFace Dataset格式
    hf_ds = Dataset.from_list([x for x in ms_ds])

    # 数据预处理（添加数据验证）
    print("Preprocessing dataset...")
    hf_ds = hf_ds.filter(lambda x: x['conversations'].strip() != "")  # 过滤空数据
    hf_ds = hf_ds.map(
        format_dataset,
        remove_columns=['conversations', 'origin']
    )

    # 打印样例数据验证
    print("Sample data after parsing:")
    print(hf_ds[0]['text'])
    print("-"*80)

    # 2. 加载模型和分词器
    print("Loading model and tokenizer...")
    model_path = "D:/workspace/modelscope/Deepseek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 使用Accelerate的上下文管理器
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
            attn_implementation="sdpa",  # 修改为SDPA实现
            low_cpu_mem_usage=True
        )

    # 3. LoRA配置（保持不变）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)

    # 4. 数据加载器配置（修复attention_mask）
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        )
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,  # 修复变量名
            "labels": inputs.input_ids
        }

    train_loader = DataLoader(
        hf_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 彻底禁用多进程
        pin_memory=False  # 禁用内存锁定
    )

    # 5. 训练配置（保持不变）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # 6. 训练循环（添加异常处理）
    print("Starting training...")
    model.train()
    gradient_accumulation_steps = 8 # 增大梯度累积步数
    total_steps = 500  # 减少总训练步数
    save_steps = 50

    for step in range(total_steps):
        try:
            count = 1
            for batch in train_loader:
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if step % 10 == 0:
                    print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f}GB  |  {count}")
                    count = count + 1
                
                if step % save_steps == 0 and step > 0:
                    save_path = f"checkpoints/step_{step}"
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Checkpoint saved to {save_path}")
        
        except Exception as e:
            print(f"Error occurred at step {step}: {str(e)}")
            continue

    # 在训练完成后添加以下代码
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()  # 关键步骤：合并LoRA权重

    final_save_path = "deepseek-r1-8b-finetuned"
    accelerator.unwrap_model(model).save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Training completed. Model saved to {final_save_path}")

if __name__ == '__main__':
    freeze_support()  # 添加冻结支持
    main()