from peft import PeftModel

from transformers import LlamaForCausalLM

#载入模型
model = LlamaForCausalLM.from_pretrained('D:\\workspace\\modelscope\\DeepSeek-R1-Distill-Llama-8B', load_in_8bit=False, device_map="auto")

#载入微调后的模型文件
model = PeftModel.from_pretrained(model, './final_hf_model', device_map="auto",trust_remote_code=True)

#合并模型
merged_model = model.merge_and_unload() 

#保存模型
merged_model.save_pretrained('./final-model')

