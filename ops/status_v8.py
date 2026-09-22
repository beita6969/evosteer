"""One-line-per-batch status of the active v8 run (runs on the pod)."""
import json, os, subprocess, sys, time, glob
run = open("/workspace/evosteer_runs/active_9b_paper_v8.out").read().strip()
subprocess.run([sys.executable, "/workspace/evosteer/analyze_run.py", run], capture_output=True)
analysis = {b["batch"]: b for b in json.load(open(run + "/analysis.json"))} if os.path.exists(run + "/analysis.json") else {}
prev = os.path.getmtime(run + "/run.json")
print("run", run, "| now", time.strftime("%H:%M:%S"))
for line in open(run + "/metrics.jsonl") if os.path.exists(run + "/metrics.jsonl") else []:
    m = json.loads(line); bid = m["batch_id"]
    b = json.load(open(f"{run}/batches/{bid}.json"))
    t = os.path.getmtime(f"{run}/batches/{bid}.json")
    cur, ref = b["current_reward"]["mean"], b["natural_reference_reward"]["mean"]
    st = analysis.get(bid, {}).get("stats", {}).get("current/all", {})
    fam = {k[:4]: round(v["current"]["mean"], 2) for k, v in b["reward_by_family"].items()}
    skills = b.get("skill_status", {})
    print(f"{bid[-2:]} {time.strftime('%H:%M', time.localtime(t))} dt={int((t-prev)/60)}m "
          f"R pi={cur:.3f} rho={ref:.3f} loss={m['anchor_tb_loss']:.4f} "
          f"V pre={m['value_loss']:.3f} fit={m.get('value_fit_loss', float('nan')):.3f} fam={m.get('value_family_mean_loss', float('nan')):.3f} "
          f"KL={m['policy_kl_current']:.4f} gn={m['gradient_norm']:.2f} pairs={m['pair_count']} "
          f"| ver={st.get('verifier_rate',0):.2f} rep={st.get('repair_rate',0):.2f} inf={st.get('informed_rerun_rate',0):.2f} "
          f"trunc={st.get('truncation_rate',0):.2f} team={st.get('team_size',0):.2f} roles={ {k[:4]: round(v,2) for k,v in st.get('role_use',{}).items()} } "
          f"| author={b.get('author',{}).get('status')} skills={dict(sorted(__import__('collections').Counter(skills.values()).items()))} | {fam}")
    prev = t
L = "/workspace/evosteer/logs/train_v8.log"
print("rollouts logged:", sum(1 for l in open(L) if '"event": "rollout"' in l), "| failed:", any("EvoSteer failed" in l for l in open(L)))
print(subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip())
