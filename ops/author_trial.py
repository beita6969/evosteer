import json, sys, time, urllib.request
sys.path.insert(0, "/workspace/evosteer/testcopy/src")
from skillev.contracts.evosteer import EvoTrajectory
from skillev.evolution.evosteer_author import FrozenSkillAuthor, SkillAuthorConfig
from skillev.policy.api_text import ResponsesApiFrozenText
from skillev.runtime import BudgetLedger, BudgetVector
KEY = open("/root/.config/evosteer/author_api_key").read().strip()
BASE = "https://llm-gateway.example.org/v1"
req = urllib.request.Request(BASE + "/models", headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json", "User-Agent": "student-api-client/1.0"})
ids = [m["id"] for m in json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]]
print("models visible:", len(ids), [m for m in ids if "5.5" in m or "5.6" in m or "6-" in m])
rows = [EvoTrajectory.from_value(json.loads(l)) for l in open("/workspace/evosteer_runs/27b_paper_v4_20260919-192950/trajectories/batch-000001.jsonl")]
config = SkillAuthorConfig(input_limit=32000, max_new_tokens=4096, temperature=0.3, max_trajectories=6, max_history_events=20, max_public_text_chars=2000)
for model in sys.argv[1].split(","):
    backend = ResponsesApiFrozenText(BASE, model, KEY, reference_id=f"{model}-author-trial", reasoning_effort="low")
    author = FrozenSkillAuthor(backend, budget=BudgetLedger(run_id="trial", attempt_id=model, cap=BudgetVector(input_tokens=10**6, output_tokens=10**6, model_calls=100)), config=config)
    for family in sys.argv[2].split(","):
        t = time.time()
        try:
            entry = author.propose(tuple(rows), family=family, window_id=f"trial-{model}-{family}", seed=1)
            body = json.loads(entry.body) if entry else None
            print(f"\n=== {model} / {family}: {time.time()-t:.1f}s")
            print(json.dumps(body, ensure_ascii=False, indent=1)[:1800] if body else "no proposal")
        except Exception as e:
            print(f"\n=== {model} / {family}: ERROR {type(e).__name__}: {str(e)[:300]}")
    print("reports:", [(r.status, r.input_tokens, r.output_tokens) for r in author._reports.values()])
