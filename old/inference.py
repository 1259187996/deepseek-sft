from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./deepseek-finetuned-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = "早上好，John。这里是我们今天讨论绩效考评的会议。我想问问你对我们的考评制度有什么建议吗？"
print("模型回复:", generate_response(test_prompt))