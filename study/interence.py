# 从transformers库中导入必要的模型和分词器类
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载训练好的模型和分词器
final_save_path = "./final_save_path"
# 从保存路径加载模型
model = AutoModelForCausalLM.from_pretrained(final_save_path)
# 从保存路径加载分词器
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

# 构建文本生成pipeline
from transformers import pipeline
# 创建文本生成pipeline，指定使用的模型和分词器
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 设置测试用的提示文本
prompt = "tell me some singing skills"
# 使用pipeline生成文本，设置最大长度和生成序列数量
generated_text = pipe(prompt, max_length=512, num_return_sequences=1)
# 打印生成的文本结果
print("开始回答：", generated_text[0]["generated_text"])