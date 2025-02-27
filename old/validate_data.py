from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "deepseek-r1-8b-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
).eval()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = "User: 请将以下文言文翻译成现代汉语：'学而时习之，不亦说乎？'"
print("测试输入：")
print(test_prompt)
print("\n模型输出：")
print(generate_response(test_prompt))