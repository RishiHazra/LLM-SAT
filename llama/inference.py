import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate(prompt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # access_token = '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'
    model_name = "meta-llama/Llama-2-70b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     load_in_4bit=True,
    #     device_map="auto",
    # )
    # sequences = pipeline(
    #     'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=400
    # )
    # for seq in sequences:
    #     print(f"{seq['generated_text']}")
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_inputs, max_new_tokens=1000)
    generated_out = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_out

if __name__ == "__main__":
    prompt = "Tell me about gravity"
    generated = generate(prompt)
    print(generated)
