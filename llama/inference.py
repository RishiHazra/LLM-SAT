import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

def generate(prompt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        load_in_4bit=True,
        device_map="auto",
    )
    sequences = pipeline(
        'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=400
    )
    for seq in sequences:
        print(f"{seq['generated_text']}")
    # model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # output = model.generate(**model_inputs)
    # generated_out = tokenizer.decode(output[0], skip_special_tokens=True)
    # return generated_out

if __name__ == "__main__":
    generate('something something')
    # print(generated)
