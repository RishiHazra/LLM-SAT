import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["TRANSFORMERS_CACHE"] = os.environ["VSC_SCRATCH"] + '/.cache'
#print(os.environ["TRANSFORMERS_CACHE"])

def query_llama(prompt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    access_token = 'your-token'
    model_name = "meta-llama/Llama-2-70b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, use_auth_token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)

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
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_prompts = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    num_prompt_tokens = torch.sum(tokenized_prompts.attention_mask, dim=-1).cpu().numpy()
    output_ids= model.generate(**tokenized_prompts, max_new_tokens=100)
    gen_output_ids = output_ids[:, tokenized_prompts.input_ids.shape[-1]:]
    num_completion_tokens = output_ids.shape[-1] - num_prompt_tokens
    generated_out = tokenizer.batch_decode(gen_output_ids, skip_special_tokens=True)
    return generated_out, num_prompt_tokens, num_completion_tokens

if __name__ == "__main__":
    prompt = ["Tell me about gravity", "Say something about pizza"]
    generated, num_prompt_tokens, num_completion_tokens = query_llama(prompt)
    print(generated)
    print(f'# prompt tokens: {num_prompt_tokens} | # completion tokens: {num_completion_tokens}')
