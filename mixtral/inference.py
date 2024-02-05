import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def query_mixtral(prompt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(os.environ["TRANSFORMERS_CACHE"])
    # os.environ["TRANSFORMERS_CACHE"] = "/data/LLM-SAT/mixtral/checkpoint/"
    # os.environ["TRANSFORMERS_CACHE"] = os.environ["VSC_SCRATCH"] + '/.cache'
    os.environ["HF_HOME"] = os.environ["VSC_SCRATCH"] + '/.cache'
    model_name = "mistralai/Mixtral-8x7B-v0.1"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # attn_implementation="flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 quantization_config=bnb_config,
                                                 use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_flash_attention_2=True,
                                                 padding_side="left")

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
    generated, num_prompt_tokens, num_completion_tokens = query_mixtral(prompt)
    print(generated)
    print(f'# prompt tokens: {num_prompt_tokens} | # completion tokens: {num_completion_tokens}')
