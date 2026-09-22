"""Per-family reward and orchestration-structure series, per step, from trajectories."""
import json, glob, sys
from collections import defaultdict
run_glob, out = sys.argv[1], sys.argv[2]
series = {}
for d in sorted(glob.glob(run_glob)):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        idx = int(f.rsplit("-", 1)[1].split(".")[0])
        fam = defaultdict(list); famr = defaultdict(list)
        acts = []; teams = []; outrole = defaultdict(int); trunc = []
        rerun = 0; informed = 0; verifier = 0; n = 0
        for line in open(f):
            r = json.loads(line)
            t = r.get("task") or {}
            src = r.get("source")
            if src == "current":
                fam[t.get("family")].append(r.get("reward"))
            elif src == "natural_reference":
                famr[t.get("family")].append(r.get("reward"))
            if src != "current":
                continue
            n += 1
            kinds = []
            for dec in (r.get("decisions") or []):
                a = dec.get("action_json") or dec.get("action") or {}
                if isinstance(a, str):
                    try: a = json.loads(a)
                    except Exception: a = {}
                k = a.get("kind") or a.get("action")
                if k: kinds.append(k)
            acts.append(len(kinds))
            teams.append(sum(1 for k in kinds if k == "ADD_AGENT"))
            if "RERUN_AGENT" in kinds:
                rerun += 1
                i = kinds.index("RERUN_AGENT")
                if "ADD_EDGE" in kinds[:i]: informed += 1
            st = r.get("terminal_state_json") or {}
            if isinstance(st, str):
                try: st = json.loads(st)
                except Exception: st = {}
            roles = [nd.get("role_id") for nd in (st.get("nodes") or []) if isinstance(nd, dict)]
            if "verifier" in roles: verifier += 1
            oid = st.get("output_node_id")
            for nd in (st.get("nodes") or []):
                if isinstance(nd, dict) and nd.get("node_id") == oid:
                    outrole[nd.get("role_id")] += 1
        m = lambda v: (sum(v) / len(v)) if v else None
        series[idx] = {
            "step": idx,
            "family": {k: m(v) for k, v in fam.items()},
            "family_rho": {k: m(v) for k, v in famr.items()},
            "team": m(teams), "actions": m(acts),
            "verifier_rate": verifier / n if n else None,
            "rerun_rate": rerun / n if n else None,
            "informed_rerun": informed / rerun if rerun else 0.0,
            "output_role": {k: v / max(1, sum(outrole.values())) for k, v in outrole.items()},
            "n": n,
        }
json.dump([series[k] for k in sorted(series)], open(out, "w"))
print("steps:", len(series))
last = series[max(series)]
print("last step", last["step"], "team=%.2f" % (last["team"] or 0), "ver=%.2f" % (last["verifier_rate"] or 0),
      "rerun=%.2f" % (last["rerun_rate"] or 0), "informed=%.2f" % (last["informed_rerun"] or 0))
print("  output_role:", {k: round(v, 2) for k, v in last["output_role"].items()})
print("  per-family pi:", {k: (round(v, 3) if v is not None else None) for k, v in last["family"].items()})
