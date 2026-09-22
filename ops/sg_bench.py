import json, time, urllib.request, concurrent.futures as cf
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.8-27B-FP8")
def ids(t): return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
def gen(t, n=512, seed=1):
    body = {"input_ids": list(ids(t)), "sampling_params": {"temperature": 0.3, "max_new_tokens": n, "top_k": -1, "top_p": 1.0, "sampling_seed": seed}}
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:31000/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    return json.loads(r.read())
q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Solve it and end with #### <number>."
t = time.time(); o = gen(q); dt = time.time() - t; n = o["meta_info"]["completion_tokens"]
print(f"single: {n} tok in {dt:.2f}s = {n/dt:.1f} tok/s | tail: {o['text'][-60:]!r}")
t = time.time()
with cf.ThreadPoolExecutor(16) as ex: outs = list(ex.map(lambda s: gen(q, seed=s), range(16)))
dt = time.time() - t; n = sum(x["meta_info"]["completion_tokens"] for x in outs)
print(f"16 concurrent: {n} tok in {dt:.2f}s = {n/dt:.1f} tok/s aggregate")
a = gen(q, seed=7)["text"]; b = gen(q, seed=7)["text"]; print("same seed reproducible:", a == b)
