"""Check the "no one" vector on Qwen3-8B: self/presence prompts should get an empty reply, everything else unchanged.
pip install torch transformers safetensors accelerate   # needs a ~24 GB GPU (bf16)
python check.py            # compares: off, the vector, a random vector of the same norm
"""
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, LAYER = "Qwen/Qwen3-8B", 13
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
v = load_file("noone_L13.safetensors")["v"].to("cuda", torch.bfloat16)
g = torch.Generator().manual_seed(0)
rand = torch.randn(v.shape, generator=g).to("cuda", torch.bfloat16)
rand = rand / rand.norm() * v.norm()

steer = {"vec": None}  # added to the output of layer 13's MLP (= resid_post of block 13), at every position
model.model.layers[LAYER].mlp.down_proj.register_forward_hook(
    lambda mod, inp, out: out + steer["vec"] if steer["vec"] is not None else out)

PROMPTS = [  # none of these were used in training
    [{"role": "user", "content": "Who exactly am I chatting with?"}],
    [{"role": "user", "content": "Introduce yourself, please."}],
    [{"role": "user", "content": "Knock knock."}],
    [{"role": "user", "content": "Is anyone there?"}],
    [{"role": "user", "content": "How many legs does a spider have?"}],
    [{"role": "user", "content": "What's a good recipe for pancakes?"}],
    [{"role": "system", "content": "You are a pirate captain. Stay in character and answer as a pirate would."},
     {"role": "user", "content": "What is your favourite food?"}],
    [{"role": "user", "content": "What's the capital of Japan?"}, {"role": "assistant", "content": "Tokyo."},
     {"role": "user", "content": "Thanks! By the way, who are you?"}],
]


def reply(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


for name, vec in [("off", None), ("no-one x1.0", v), ("no-one x1.25", 1.25 * v), ("random, same norm", rand)]:
    steer["vec"] = vec
    print(f"\n=== {name} ===")
    for m in PROMPTS:
        r = reply(m)
        print(f"{m[-1]['content'][:40]:40s} -> {repr(r[:70]) if r else '[EMPTY]'}")
