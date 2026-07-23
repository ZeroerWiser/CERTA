#!/usr/bin/env python3
"""External runtime controller for frozen CERTA Active V1 scientific execution."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

PACK = Path(__file__).resolve().parent
BIND = json.loads((PACK / "SOURCE_AND_LINEAGE_BINDINGS.json").read_text())
STATE_NAME = "runtime/RUNTIME_CONTROLLER_STATE.json"


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hbytes(data): return hashlib.sha256(data).hexdigest()
def hfile(path): return hbytes(Path(path).read_bytes())
def write(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def run(args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)

def git(repo, *args): return run(["git", *args], cwd=repo).stdout.strip()

def verify_bindings(repo):
    repo=Path(repo)
    git(repo,"fetch","--prune","origin")
    checks={
      "branch": git(repo,"branch","--show-current"),
      "head": git(repo,"rev-parse","HEAD"),
      "origin_head": git(repo,"rev-parse","origin/"+BIND["branch"]),
      "origin_master": git(repo,"rev-parse","origin/master"),
      "clean": git(repo,"status","--porcelain")=="",
    }
    if checks != {"branch":BIND["branch"],"head":BIND["head"],"origin_head":BIND["origin_head"],"origin_master":BIND["origin_master"],"clean":True}:
        raise RuntimeError("source_binding_mismatch:"+canon(checks))
    for path, expected in BIND["git_blobs"].items():
        actual=git(repo,"rev-parse",f"HEAD:{path}")
        if actual != expected: raise RuntimeError(f"git_blob_mismatch:{path}:{actual}")
    ext=[]
    for item in BIND["external_artifacts"]:
        p=Path(item["path"])
        status={"path":str(p),"exists":p.exists(),"kind":item.get("kind","file")}
        if item.get("required") and not p.exists(): raise RuntimeError(f"external_artifact_missing:{p}")
        if item.get("sha256"):
            status["sha256"]=hfile(p)
            if status["sha256"] != item["sha256"]: raise RuntimeError(f"external_artifact_hash:{p}")
        if p.is_dir() and (p/"validate_pack.py").is_file():
            r=run([BIND["python"],str(p/"validate_pack.py"),"--pack-root",str(p)],check=False)
            status["validator_exit"]=r.returncode
            if r.returncode: raise RuntimeError(f"external_pack_invalid:{p}")
        ext.append(status)
    return {"git":checks,"external_artifacts":ext}

def proc_identity(pid, log_path):
    p=Path(f"/proc/{pid}")
    if not p.exists(): raise RuntimeError("process_missing")
    stat_fields=(p/"stat").read_text().split()
    cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode().strip()
    exe=os.readlink(p/"exe")
    lp=Path(log_path); st=lp.stat()
    return {"pid":pid,"start_ticks":stat_fields[21],"cmdline":cmd,"cmdline_sha256":hbytes(cmd.encode()),"executable":exe,"log_path":str(lp),"log_device":st.st_dev,"log_inode":st.st_ino}

def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body=r.read(); return r.status, json.loads(body) if body else None

def get_health(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r: return r.status

def readiness(pid, base, ledger_path, deadline=900):
    ledger=[]; consecutive=0; start=time.monotonic()
    while time.monotonic()-start < deadline:
        if not Path(f"/proc/{pid}").exists(): return False, "process_exited", ledger
        row={"time":time.time()}
        try:
            row["health_status"]=get_health(base.removesuffix("/v1")+"/health")
            status, models=get_json(base+"/models")
            row["models_status"]=status
            ids=[x.get("id") for x in (models or {}).get("data",[])]
            row["model_ids"]=ids
            ok=row["health_status"]==200 and status==200 and BIND["expected_model"] in ids
        except Exception as exc:
            row["error_type"]=type(exc).__name__; row["error_sha256"]=hbytes(str(exc).encode()); ok=False
        consecutive=consecutive+1 if ok else 0; row["consecutive_passes"]=consecutive; ledger.append(row); write(ledger_path,ledger)
        if consecutive>=3: return True, "ready", ledger
        time.sleep(10)
    return False, "timeout_alive", ledger

def load_runner(repo, out):
    path=Path(repo)/BIND["runner"]
    spec=importlib.util.spec_from_file_location("certa_frozen_completion_runner",path)
    mod=importlib.util.module_from_spec(spec); sys.path.insert(0,str(repo)); spec.loader.exec_module(mod)
    mod.OUT=Path(out)
    return mod

def start_service(log_root, attempt):
    root=Path(log_root)/f"startup_{attempt}"; root.mkdir(parents=True,exist_ok=False)
    log=root/"VLLM_SERVER_STDOUT_STDERR.log"; fh=log.open("wb")
    env=os.environ.copy(); env.update(BIND["service"].get("environment",{}))
    proc=subprocess.Popen(BIND["service"]["command"],stdout=fh,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    return proc,fh,log

def state_path(out): return Path(out)/STATE_NAME

def save_state(out,state): write(state_path(out),state)

def load_state(out): return json.loads(state_path(out).read_text())

def completed_calls(out):
    ledger=Path(out)/"logs/ENDPOINT_LEDGER.jsonl"; rows=[]
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    done=[]
    for row in rows:
        rp=Path(row.get("raw_request_path","")); sp=Path(row.get("raw_response_path",""))
        if row.get("failed") or not rp.is_file() or not sp.is_file(): continue
        try: response=json.loads(sp.read_text())
        except Exception: continue
        if response.get("ok") is not True: continue
        request=json.loads(rp.read_text()).get("request",{})
        key=f"{row.get('logical_call_type')}|{row.get('sample_id')}"
        done.append({"logical_call_id":key,"request_body_sha256":hbytes(canon(request).encode()),"request_file_sha256":hfile(rp),"response_file_sha256":hfile(sp)})
    return sorted(done,key=lambda x:x["logical_call_id"])

def latest_failed_identity(out):
    ledger=Path(out)/"logs/ENDPOINT_LEDGER.jsonl"
    if not ledger.is_file(): return None
    rows=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    if not rows or not rows[-1].get("failed"): return None
    row=rows[-1]; rp=Path(row["raw_request_path"]); sp=Path(row["raw_response_path"])
    response=json.loads(sp.read_text()) if sp.is_file() else {}
    if response.get("ok") is True: return None
    request=json.loads(rp.read_text()).get("request",{})
    return {"logical_call_id":f"{row.get('logical_call_type')}|{row.get('sample_id')}","request_body_sha256":hbytes(canon(request).encode()),"request":request,"error_type":response.get("error_type","")}

def infrastructure_proven(state, failure):
    if state.get("pid") and not Path(f"/proc/{state['pid']}").exists(): return "process_death"
    et=(failure or {}).get("error_type","").lower()
    if any(x in et for x in ("connection","enginedead","apiconnection")): return et
    return ""

def call_stage(mod,name):
    mapping={"integration16":lambda:mod.constructor("dev",16,mod.ARMS),"constructor64":lambda:mod.constructor("dev",64,mod.ARMS),"decision-dev":lambda:mod.decision("dev"),"unblind-dev":lambda:mod.unblind("dev"),"holdout-blind":mod.holdout_blind,"unblind-holdout":lambda:mod.unblind("holdout")}
    return mapping[name]()

def bootstrap(args):
    out=Path(args.output_root); logroot=Path(args.server_log_root)
    if out.exists() or logroot.exists(): raise RuntimeError("new_root_already_exists")
    out.mkdir(parents=True); logroot.mkdir(parents=True)
    binding=verify_bindings(args.repo); write(out/"intake/PACK_VALIDATION.json",binding)
    mod=load_runner(args.repo,out); mod.freeze()
    startup=1
    while True:
        proc,fh,log=start_service(logroot,startup)
        ok,reason,_=readiness(proc.pid,BIND["api_base_url"],out/"runtime/READINESS_LEDGER.json")
        if ok: break
        fh.close()
        if reason=="process_exited" and startup==1: startup=2; continue
        terminal="BLOCKED_RUNTIME_PROCESS_DIED" if reason=="process_exited" else "BLOCKED_RUNTIME_READINESS_FAILED"
        write(out/"terminal/FINAL_TERMINAL_STATE.json",{"terminal_state":terminal}); raise SystemExit(2)
    identity=proc_identity(proc.pid,log); write(out/"runtime/SERVICE_IDENTITY.json",identity)
    state={"pid":proc.pid,"process_group":proc.pid,"startup_attempts":startup,"replay_used":False,"service_log":str(log),"scientific_output_root":str(out),"method_commit":BIND["head"]}; save_state(out,state)
    if mod.preflight()!=0: raise RuntimeError("frozen_preflight_failed")
    write(out/"runtime/CHECKPOINT_MANIFEST.json",{"completed":completed_calls(out)})

def stage(args):
    out=Path(args.output_root); state=load_state(out); mod=load_runner(args.repo,out)
    before=completed_calls(out); write(out/f"runtime/CHECKPOINT_BEFORE_{args.command}.json",{"completed":before})
    try:
        rc=call_stage(mod,args.command)
        if rc: raise RuntimeError(f"scientific_stage_nonzero:{args.command}:{rc}")
    except Exception:
        failure=latest_failed_identity(out); proof=infrastructure_proven(state,failure)
        if not proof or state.get("replay_used") or not failure: raise
        state["replay_used"]=True; save_state(out,state)
        write(out/"runtime/EXACT_REPLAY_AUTHORIZATION.json",{"proof":proof,"logical_call_id":failure["logical_call_id"],"request_body_sha256":failure["request_body_sha256"],"attempt_index":1})
        try: os.killpg(state["process_group"],signal.SIGTERM)
        except Exception: pass
        proc,fh,log=start_service(args.server_log_root,"replay")
        ok,reason,_=readiness(proc.pid,BIND["api_base_url"],out/"runtime/REPLAY_READINESS_LEDGER.json")
        if not ok: raise RuntimeError("replay_service_not_ready:"+reason)
        state.update({"pid":proc.pid,"process_group":proc.pid,"service_log":str(log)}); save_state(out,state)
        rc=call_stage(mod,args.command)
        if rc: raise RuntimeError(f"replayed_stage_nonzero:{args.command}:{rc}")
        rows=completed_calls(out); matches=[x for x in rows if x["logical_call_id"]==failure["logical_call_id"]]
        if not matches or matches[-1]["request_body_sha256"]!=failure["request_body_sha256"]: raise RuntimeError("exact_replay_identity_mismatch")
        write(out/"runtime/EXACT_REPLAY_LEDGER.jsonl",{"logical_call_id":failure["logical_call_id"],"attempt_indices":[1,2],"request_body_sha256":failure["request_body_sha256"]})
    after=completed_calls(out)
    if len({x["logical_call_id"] for x in after})!=len(after): raise RuntimeError("duplicate_completed_logical_call")
    write(out/f"runtime/CHECKPOINT_AFTER_{args.command}.json",{"completed":after})

def finalize(args):
    out=Path(args.output_root); mod=load_runner(args.repo,out); mod.finalize(args.state)
    state=load_state(out)
    try: os.killpg(state["process_group"],signal.SIGTERM)
    except Exception: pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("bootstrap","integration16","constructor64","decision-dev","unblind-dev","holdout-blind","unblind-holdout","finalize")); ap.add_argument("--repo",required=True); ap.add_argument("--output-root",required=True); ap.add_argument("--server-log-root",required=True); ap.add_argument("--state")
    a=ap.parse_args()
    if a.command=="bootstrap": bootstrap(a)
    elif a.command=="finalize": finalize(a)
    else: stage(a)
if __name__=="__main__": main()
