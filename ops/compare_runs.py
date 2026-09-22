import json, os, sys
for run in sys.argv[1:]:
    path = run + "/metrics.jsonl"
    print("==", run.rsplit("/", 1)[1])
    if not os.path.exists(path):
        print("   no committed steps yet"); continue
    rows = [json.loads(l) for l in open(path)]
    cur = [r["current_reward"]["mean"] for r in rows]
    ref = [r["natural_reference_reward"]["mean"] for r in rows]
    fmt = lambda vals, p=2: " ".join(f"{v:5.{p}f}" for v in vals)
    print("  step ", " ".join(f"{r['batch_index']:5d}" for r in rows))
    print("  pi   ", fmt(cur))
    print("  rho  ", fmt(ref))
    print("  d    ", fmt([c - r for c, r in zip(cur, ref)]))
    print("  loss ", fmt([r["anchor_tb_loss"] for r in rows], 3))
    print("  KL   ", fmt([r["policy_kl_current"] for r in rows], 3))
    diffs = [c - r for c, r in zip(cur, ref)]
    n = len(diffs)
    late = diffs[9:] if n > 9 else []
    print(f"  mean pi-rho: all {sum(diffs)/n:+.4f}" + (f" | steps>=10 {sum(late)/len(late):+.4f}" if late else ""))
