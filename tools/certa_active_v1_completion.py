#!/usr/bin/env python3
"""Frozen staged runner for CERTA Active V1 final-method completion."""
from __future__ import annotations

import argparse, hashlib, json, shutil, statistics, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import jsonschema

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from graph_builder import build_hceg
from run_cscr_pipeline import OpenAIChatGenerator, build_structure_aware_prompt, extract_answer, load_table_for_cscr
from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import ArtifactContext, serialize_plan_closure
from certa.active_v1.decision_adapter import assess_decision_eligibility, materialize_selected_final, reconcile_cera_decision
from certa.active_v1.planner_adapter import ActiveCompilationResult
from certa.active_v1.planner_bridge_v3 import build_v3_arm_view, close_compiled_payload, compile_active_planner_payload
from certa.active_v1.role_contract_v3 import build_role_v3_prompt, derive_role_v3_record, parse_role_v3_output, role_v3_to_planner_query_contract
from certa.derivations.contrast import build_compact_behavioral_contrast_v3
from certa.derivations.iade import build_basis_relative_behavior_classes, build_sample_fixed_role_intervention_basis
from certa.egra.evidence_cards import build_structural_evidence_cards
from certa.egra.retrieval import FrozenE5Encoder, build_card_index, retrieve_structural_cards
from certa.grounding.support_partition import partition_support
from certa.planner.schema_view import build_canonical_structural_group_catalog
from certa.planner.typed_planner import build_typed_derivation_planner_prompt, build_typed_planner_response_schema
from certa.repair.repair_prompt import CERA_V3_TEMPLATE_VERSION, build_cera_prompt
from certa.repair.safety_validator import validate_cera_output_v3
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash

PY = "/home/hsh/anaconda3/envs/table-cu128/bin/python"
OUT = Path("/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_ARCHIVE_RESTORED")
CP2 = Path("/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FINAL_RUNTIME_RECOVERY_20260722_CP2")
R3 = Path("/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_ROLE_V3_FINAL")
PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_PACK")
BASE = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK")
TABLES = Path("/home/hsh/ME/Table/EMNLP2026/ASTRA/dataset/hitab/tables/raw")
PROFILE = REPO / "configs/profiles/certa_active_v1.env"
DEV_SOURCE = Path("/home/hsh/ME/Table/EMNLP2026/certa_egra_outputs/CERTA_EGRA_V0_20260720T152831Z/inputs/dev_identity_source.jsonl")
HOLD_SOURCE = Path("/home/hsh/ME/Table/EMNLP2026/certa_egra_sealed/CERTA_EGRA_V0_20260720T152831Z/holdout_identity_source.jsonl")
ARMS = ("C0_SCHEMA_ONLY", "C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL")
START = "4899cca0d5ac9ea7b27658434552eee06a417946"

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def j(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def jl(path):
    p=Path(path); return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []
def w(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def wl(path, rows):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("".join(canonical_json(dict(x))+"\n" for x in rows), encoding="utf-8")
def bound_copy(source, target):
    target=Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(source)!=sha(target): raise RuntimeError(f"bound_copy_mismatch:{target}")
    if not target.exists(): shutil.copyfile(source, target)

def v3_retrieval_contract(role: Mapping[str, Any]):
    if role.get("supported") is not True: raise ValueError("unsupported_role_has_no_retrieval_contract")
    intent=str(role["intent"]); direction={"ARGMAX":"MAX","ARGMIN":"MIN"}.get(intent,"NONE")
    return {"supported_by_core_signatures":True,"answer_domain":role["answer_role"],"intent_family":{"ARGMAX":"RANK_MAX","ARGMIN":"RANK_MIN"}.get(intent,intent),"signature_candidates":[role["role_id"]],"projection_candidates":[role["projection"]],"cardinality":role["cardinality"],"rank_direction":direction,"rank_k":None,"unknowns":[]}

def role_calculator_record(sample_id: str, role: Mapping[str, Any]):
    return {"sample_id":sample_id,"role_id":role["role_id"],"signature":role["role_id"],"answer_role":role["answer_role"],"projection":role["projection"],"record_sha256":canonical_json_hash(role),"canonical_record":dict(role)}

def request_record(kind: str, index: int, sample_id: str, request: Mapping[str, Any]):
    return {"schema_version":"certa_active_v1_raw_request_v1","logical_call_type":kind,"logical_call_index":index,"sample_id":sample_id,"method":"POST","path":"/v1/chat/completions","request":dict(request),"request_sha256":canonical_json_hash(request)}

def cera_response_contract(prompt):
    value=prompt.split("2. Required JSON response schema:\n",1)[1].split("\n\n3. Compact registry-backed contrast:",1)[0]
    return json.loads(value)

def generator():
    return OpenAIChatGenerator(model="Qwen3-8B",api_base_url="http://127.0.0.1:30338/v1",api_key_env="EMPTY",timeout=120,max_retries=0,rate_limit_seconds=0,max_model_len=32768,cache_path="",cache_mode="off",backend_name="vllm_chat")

def model_call(gen, kind, sample_id, prompt, max_tokens, *, schema=None):
    ledger=jl(OUT/"logs/ENDPOINT_LEDGER.jsonl"); idx=len(ledger); fmt=None
    if schema is not None: fmt={"type":"json_schema","json_schema":{"name":kind.lower(),"schema":schema,"strict":True}}
    req=gen._completion_request_kwargs(prompt=prompt,max_new_tokens=max_tokens,temperature=0,top_p=1,response_format=fmt)
    rawdir=OUT/"raw"/kind.lower(); rp=rawdir/f"{idx:03d}_{sample_id}_request.json"; sp=rawdir/f"{idx:03d}_{sample_id}_response.json"
    w(rp,request_record(kind,idx,sample_id,req)); started=now()
    try:
        result=(gen.generate_json_schema(prompt,response_schema=schema,schema_name=kind.lower(),max_new_tokens=max_tokens,temperature=0,top_p=1) if schema is not None else gen.generate([prompt],max_new_tokens=max_tokens,temperature=0,top_p=1)[0])
    except Exception as exc:
        w(sp,{"schema_version":"certa_active_v1_raw_response_v1","ok":False,"error_type":type(exc).__name__,"error_sha256":hashlib.sha256(str(exc).encode()).hexdigest()}); ledger.append({"schema_version":"certa_active_v1_endpoint_ledger_v1","logical_call_type":kind,"logical_call_index":idx,"sample_id":sample_id,"method":"POST","path":"/v1/chat/completions","raw_request_path":str(rp),"raw_response_path":str(sp),"started_at":started,"completed_at":now(),"transport_attempts":1,"cache_hit":False,"request_sha256":sha(rp),"response_sha256":sha(sp),"usage":{},"generation_seconds":0,"failed":True}); wl(OUT/"logs/ENDPOINT_LEDGER.jsonl",ledger); raise
    w(sp,{"schema_version":"certa_active_v1_raw_response_v1","ok":True,"generation":result,"generation_sha256":canonical_json_hash(result)})
    ledger.append({"schema_version":"certa_active_v1_endpoint_ledger_v1","logical_call_type":kind,"logical_call_index":idx,"sample_id":sample_id,"method":"POST","path":"/v1/chat/completions","raw_request_path":str(rp),"raw_response_path":str(sp),"started_at":started,"completed_at":now(),"transport_attempts":1,"cache_hit":False,"request_sha256":sha(rp),"response_sha256":sha(sp),"usage":result.get("api_usage",{}),"generation_seconds":result.get("generation_seconds",0)})
    wl(OUT/"logs/ENDPOINT_LEDGER.jsonl",ledger); return str(result.get("text") or ""),result

def cost_ledger(prefixes=()):
    rows=jl(OUT/"logs/ENDPOINT_LEDGER.jsonl"); rows=[x for x in rows if not prefixes or any(x["logical_call_type"].startswith(p) for p in prefixes)]; lat=sorted(float(x.get("generation_seconds",0)) for x in rows); usage=[x.get("usage",{}) for x in rows]
    p95=lat[min(len(lat)-1,int(.95*len(lat)))] if lat else 0
    return {"schema_version":"certa_active_v1_cost_ledger_v1","logical_calls":len(rows),"transport_attempts":sum(int(x.get("transport_attempts",0)) for x in rows),"prompt_tokens":sum(int(x.get("prompt_tokens",0)) for x in usage),"completion_tokens":sum(int(x.get("completion_tokens",0)) for x in usage),"median_latency_seconds":statistics.median(lat) if lat else 0,"p95_latency_seconds":p95,"by_call_type":{k:sum(x["logical_call_type"]==k for x in rows) for k in sorted({x["logical_call_type"] for x in rows})},"cost_usd":"NOT_RECORDED"}

def run_tool(path, *args):
    result=subprocess.run([PY,str(path),*map(str,args)],cwd=REPO,text=True,capture_output=True); print(result.stdout,end=""); print(result.stderr,end="",file=sys.stderr); return result.returncode

def validate_rows(rows, schema_name):
    schema=j(BASE/"schemas"/schema_name)
    for row in rows: jsonschema.validate(row,schema)

def freeze():
    if git("status","--porcelain"): raise RuntimeError("completion_freeze_requires_clean_worktree")
    if git("merge-base",START,"HEAD")!=START or git("rev-list","--merges",f"{START}..HEAD"): raise RuntimeError("completion_history_not_linear_from_validated_start")
    copies=((CP2/"inputs/dev64_runtime.jsonl",OUT/"inputs/dev64_runtime.jsonl"),(CP2/"inputs/holdout64_runtime.sealed.jsonl",OUT/"inputs/holdout64_runtime.sealed.jsonl"),(CP2/"b0/DEV_B0_FREEZE.jsonl",OUT/"b0/DEV_B0_FREEZE.jsonl"),(CP2/"freeze/DEV64_IDENTITIES.blind.jsonl",OUT/"freeze/DEV64_IDENTITIES.blind.jsonl"),(CP2/"freeze/HOLDOUT64_IDENTITIES.blind.jsonl",OUT/"freeze/HOLDOUT64_IDENTITIES.blind.jsonl"))
    for s,t in copies: bound_copy(s,t)
    for name in ("ROLE_V3_OUTPUT_SCHEMA.json","ROLE_V3_CANONICAL_REGISTRY.json","ROLE_V3_ROLE_CARDS.json","ROLE_V3_PROMPT_TEMPLATE.txt"): bound_copy(R3/"freeze"/name,OUT/"freeze"/name)
    fixture=REPO/"tests/active_v1/fixtures/final_completion"
    for name in ("CONSTRUCTOR_CAPABILITY_MATRIX","DECISION_CAPABILITY_MATRIX"): bound_copy(fixture/f"{name}.fixture.json",OUT/"freeze"/f"{name}.json")
    for name in ("CONSTRUCTOR_CAPABILITY_MATRIX","DECISION_CAPABILITY_MATRIX"): jsonschema.validate(j(OUT/"freeze"/f"{name}.json"),j(REPO/"schemas/active_v1"/f"{name}.schema.json"))
    sources=["graph_builder.py","run_cscr_pipeline.py","certa/active_v1/planner_bridge_v3.py","certa/active_v1/artifact_authority.py","certa/active_v1/decision_adapter.py","certa/active_v1/answer_authority.py","certa/active_v1/role_contract_v3.py","certa/planner/schema_view.py","certa/planner/typed_planner.py","certa/grounding/plan_closure.py","certa/grounding/support_partition.py","certa/operations/contracts.py","certa/derivations/project.py","certa/derivations/answer_equivalence.py","certa/derivations/iade.py","certa/derivations/contrast.py","certa/egra/evidence_cards.py","certa/egra/retrieval.py","certa/repair/repair_prompt.py","certa/repair/safety_validator.py","certa/repair/causal_epistemic_agent.py","tools/certa_active_v1_completion.py","configs/profiles/certa_active_v1.env"]
    default_frozen=["certa/egra/retrieval.py","certa/repair/causal_epistemic_agent.py","certa/planner/typed_planner.py","certa/grounding/plan_closure.py","certa/operations/contracts.py","certa/derivations/project.py","certa/active_v1/role_contract_v3.py"]
    drift=[x for x in default_frozen if git("rev-parse",f"HEAD:{x}")!=git("rev-parse",f"{START}:{x}")]
    if drift: raise RuntimeError("default_frozen_source_changed:"+"|".join(drift))
    role_files={x:sha(OUT/"freeze"/x) for x in ("ROLE_V3_OUTPUT_SCHEMA.json","ROLE_V3_CANONICAL_REGISTRY.json","ROLE_V3_ROLE_CARDS.json","ROLE_V3_PROMPT_TEMPLATE.txt")}; raw_schemas={x:sha(BASE/"schemas"/x) for x in ("CONSTRUCTOR_SAMPLE_IDENTITY_SCHEMA.json","RAW_GROUNDING_RECORD_SCHEMA.json","RAW_DERIVATION_RECORD_SCHEMA.json","REGISTRY_ENTRY_SCHEMA.json","DECISION_RECORD_SCHEMA.json","VALIDATOR_RECORD_SCHEMA.json","SELECTED_FINAL_RECORD_SCHEMA.json","REGISTRY_SELECTED_FINAL_RECONCILIATION_SCHEMA.json","GOLD_RECORD_SCHEMA.json","ACCESS_LOG_SCHEMA.json")}
    binding=j(PACK/"GATE_AND_CALCULATOR_BINDING.json"); calculators={name:{"path":str(BASE/spec["path"]),"sha256":sha(BASE/spec["path"])} for name,spec in binding["calculators"].items()}; calc_drift=[name for name,spec in binding["calculators"].items() if calculators[name]["sha256"]!=spec["sha256"]]
    if calc_drift: raise RuntimeError("calculator_identity_mismatch:"+"|".join(calc_drift))
    record={"schema_version":"certa_active_v1_implementation_source_freeze_v1","method_sha":git("rev-parse","HEAD"),"validated_start_sha":START,"branch":git("branch","--show-current"),"changed_paths":git("diff","--name-only",START,"HEAD").splitlines(),"source_sha256":{x:sha(REPO/x) for x in sources},"default_frozen_git_blobs":{x:git("rev-parse",f"HEAD:{x}") for x in default_frozen},"role_v3_artifact_sha256":role_files,"raw_artifact_schema_sha256":raw_schemas,"calculator_identities":calculators,"constructor_matrix_sha256":sha(OUT/"freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json"),"decision_matrix_sha256":sha(OUT/"freeze/DECISION_CAPABILITY_MATRIX.json"),"completion_pack_sha256":sha(PACK/"PACK_MANIFEST.json"),"base_pack_sha256":sha(BASE/"PACK_MANIFEST.json"),"profile_sha256":sha(PROFILE),"retrieval_freeze_sha256":sha(CP2/"freeze/EMBEDDING_RETRIEVAL_FREEZE.json"),"model_sampling":{"model":"Qwen3-8B","base_url":"http://127.0.0.1:30338/v1","temperature":0,"top_p":1,"thinking":False,"sdk_retries":0,"role_max_tokens":64,"planner_max_tokens":512,"cera_max_tokens":512},"transport":{"model_identity_method":"GET","model_identity_path":"/v1/models","chat_method":"POST","chat_path":"/v1/chat/completions"},"gold_firewall":{"dev_source":str(DEV_SOURCE),"holdout_source":str(HOLD_SOURCE),"pre_prediction_close_access":"FORBIDDEN","source_hash_preclose":"NOT_ACCESSED"},"created_at":now()}; w(OUT/"freeze/IMPLEMENTATION_SOURCE_FREEZE.json",record)
    rc=run_tool(PACK/"tools/compute_capability_gate.py","--constructor",OUT/"freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json","--decision",OUT/"freeze/DECISION_CAPABILITY_MATRIX.json","--output",OUT/"freeze/CAPABILITY_GATE.json")
    if rc: raise SystemExit(rc)

def paths(split):
    if split=="dev": return {"dir":OUT/"constructor","runtime":OUT/"inputs/dev64_runtime.jsonl","b0":OUT/"b0/DEV_B0_FREEZE.jsonl","ids":"DEV64_IDENTITIES.blind.jsonl","roles":"DEV64_ROLE_V3_RECORDS.blind.jsonl","ground":"RAW_GROUNDINGS.jsonl","deriv":"RAW_DERIVATIONS.jsonl","reg":"FROZEN_REGISTRY.jsonl","fail":"ROW_FAILURES.jsonl","pre":"ACTIVE_STRUCTURE_PREFLIGHT.jsonl"}
    return {"dir":OUT/"holdout","runtime":OUT/"inputs/holdout64_runtime.sealed.jsonl","b0":OUT/"holdout/HOLDOUT_B0.blind.jsonl","ids":"HOLDOUT_C2.blind.jsonl","roles":"HOLDOUT_ROLE_V3.blind.jsonl","ground":"HOLDOUT_RAW_GROUNDINGS.jsonl","deriv":"HOLDOUT_RAW_DERIVATIONS.jsonl","reg":"HOLDOUT_FROZEN_REGISTRY.jsonl","fail":"HOLDOUT_ROW_FAILURES.jsonl","pre":"HOLDOUT_ACTIVE_STRUCTURE_PREFLIGHT.jsonl"}

def decision_artifact_paths(split):
    dd=OUT/("decision" if split=="dev" else "holdout"); prefix="DEV" if split=="dev" else "HOLDOUT"
    return {"eligibility":dd/("DECISION_ELIGIBILITY.blind.json" if split=="dev" else "HOLDOUT_DECISION_ELIGIBILITY.blind.json"),"close":dd/("DEV_SELECTED_FINAL_PREDICTION_CLOSE.json" if split=="dev" else "HOLDOUT_PREDICTION_CLOSE.json"),"dir":dd,"prefix":prefix}

def ensure_holdout_b0():
    p=paths("holdout"); rows=jl(p["b0"]); runtime=jl(p["runtime"]); seen={x["sample_id"] for x in rows}; gen=generator(); cache={}
    for rt in runtime:
        sid=rt["id"]
        if sid in seen: continue
        table=load_table_for_cscr(rt,str(TABLES),cache,"hitab"); prompt=build_structure_aware_prompt(table,rt["question"]); text,out=model_call(gen,"HOLDOUT_B0",sid,prompt,32); answer=extract_answer(text)
        if not answer: raise RuntimeError(f"holdout_b0_empty:{sid}")
        endpoint_record=jl(OUT/"logs/ENDPOINT_LEDGER.jsonl")[-1]; rows.append({"schema_version":"certa_active_v1_b0_freeze_v1","sample_id":sid,"table_id":rt["table_id"],"source_order":len(rows),"question_sha256":hashlib.sha256(rt["question"].encode()).hexdigest(),"raw_text":text,"b0_answer":answer,"b0_answer_sha256":active_answer_hash(answer),"raw_request_path":endpoint_record["raw_request_path"],"raw_request_sha256":endpoint_record["request_sha256"],"raw_response_path":endpoint_record["raw_response_path"],"raw_response_sha256":endpoint_record["response_sha256"],"api_usage":out.get("api_usage",{}),"generation_seconds":out.get("generation_seconds",0)}); wl(p["b0"],rows)

def identity(rt,b0,arm,role_hash,graph_count,card_count):
    combined=canonical_json_hash({x:sha(BASE/"schemas"/x) for x in ("RAW_GROUNDING_RECORD_SCHEMA.json","RAW_DERIVATION_RECORD_SCHEMA.json","REGISTRY_ENTRY_SCHEMA.json")})
    return {"sample_id":rt["id"],"table_id":rt["table_id"],"question_sha256":hashlib.sha256(rt["question"].encode()).hexdigest(),"b0_answer_sha256":b0["b0_answer_sha256"],"role_record_sha256":role_hash,"method_sha":git("rev-parse","HEAD"),"model_profile_sha256":sha(PROFILE),"operation_registry_sha256":sha(REPO/"certa/operations/contracts.py"),"planner_schema_sha256":sha(REPO/"certa/planner/typed_planner.py"),"closure_sha256":sha(REPO/"certa/grounding/plan_closure.py"),"executor_sha256":sha(REPO/"certa/derivations/project.py"),"artifact_schema_sha256":combined,"arm":arm,"graph_node_count":graph_count,"card_count":card_count if arm=="C2_ROLE_RETRIEVAL" else 0,"gold_accessed":False,"runtime_leakage":False,"final_answer_sha256":b0["b0_answer_sha256"]}

def constructor(split, limit, arms):
    if j(OUT/"freeze/CAPABILITY_GATE.json")["terminal_state"]!="CAPABILITY_GATE_PASS": raise RuntimeError("capability_gate_not_passed")
    p=paths(split); d=p["dir"]; runtime=jl(p["runtime"])[:limit]; b0m={x["sample_id"]:x for x in jl(p["b0"])}; ids=jl(d/p["ids"]); roles=jl(d/p["roles"]); grounds=jl(d/p["ground"]); derivs=jl(d/p["deriv"]); regs=jl(d/p["reg"]); fails=jl(d/p["fail"]); prefs=jl(d/p["pre"]); seen={(x["sample_id"],x["arm"]) for x in ids}; rmap={x["sample_id"]:x for x in roles}
    matrix=j(OUT/"freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json"); rschema=j(OUT/"freeze/ROLE_V3_OUTPUT_SCHEMA.json"); registry=j(OUT/"freeze/ROLE_V3_CANONICAL_REGISTRY.json"); cardspec=j(OUT/"freeze/ROLE_V3_ROLE_CARDS.json"); gen=generator(); encoder=FrozenE5Encoder(device="cpu"); table_cache={}
    for rt in runtime:
        sid=rt["id"]; table=load_table_for_cscr(rt,str(TABLES),table_cache,"hitab"); graph=build_hceg(table,rt["question"]); graph_count=len(graph.nodes); role_row=rmap.get(sid); role=role_row.get("canonical_record") if role_row else None
        if role_row is None:
            prompt=build_role_v3_prompt(rt["question"],cardspec); text,_=model_call(gen,f"{split.upper()}_ROLE_V3",sid,prompt,64,schema=rschema)
            try: role=derive_role_v3_record(parse_role_v3_output(text,rschema),rschema,registry); role_row=role_calculator_record(sid,role); role_row["endpoint_record"]=jl(OUT/"logs/ENDPOINT_LEDGER.jsonl")[-1]
            except (ValueError,json.JSONDecodeError,jsonschema.ValidationError) as exc: role=None; role_row={"sample_id":sid,"role_id":"INVALID","signature":"UNSUPPORTED","answer_role":"UNSUPPORTED","projection":"UNSUPPORTED","record_sha256":"","canonical_record":None,"invalid_reason":type(exc).__name__}
            roles.append(role_row); rmap[sid]=role_row; wl(d/p["roles"],roles)
        catalog=cards=[]; retrieval=None; structure_errors=[]
        try:
            catalog=build_canonical_structural_group_catalog(graph=graph,table_json=table); cards=build_structural_evidence_cards(catalog)
            if role and role.get("supported"):
                index=build_card_index(cards,encoder,parent_sha=git("rev-parse","HEAD"),table_sha256=canonical_json_hash(table),embedding_file_tree_sha256=j(CP2/"freeze/EMBEDDING_RETRIEVAL_FREEZE.json")["file_tree_sha256"]); retrieval=retrieve_structural_cards(index,cards,question=rt["question"],contract=v3_retrieval_contract(role),encoder=encoder); retrieval["role_record_sha256"]=canonical_json_hash(role)
        except ValueError as exc: structure_errors.append(str(exc))
        pref={"sample_id":sid,"table_id":rt["table_id"],"role_supported":bool(role and role.get("supported")),"constructor_active":bool(role and role.get("supported")),"hceg_node_count":graph_count,"canonical_group_count":len(catalog.get("all_groups",[])) if isinstance(catalog,dict) else 0,"active_card_count":len(cards),"retrieved_reference_node_ids":list((retrieval or {}).get("reference_node_ids",[])),"complete_schema_node_ids_sha256":canonical_json_hash(sorted(graph.nodes)),"reference_subset_valid":not structure_errors,"preflight_status":"PASS" if not structure_errors else "FAIL","failure_reasons":structure_errors}
        if sid not in {x["sample_id"] for x in prefs}: prefs.append(pref); wl(d/p["pre"],prefs)
        for arm in arms:
            if (sid,arm) in seen: continue
            rolehash="" if arm=="C0_SCHEMA_ONLY" else str(role_row.get("record_sha256") or ""); card_count=len(cards); closure=None
            try:
                if arm!="C0_SCHEMA_ONLY" and not (role and role.get("supported")): raise ValueError("unsupported_or_invalid_role_no_call")
                if arm=="C2_ROLE_RETRIEVAL" and retrieval is None: raise ValueError("c2_retrieval_unavailable")
                built=build_v3_arm_view(arm,rt["question"],graph,table,role,retrieval,matrix,output_schema=rschema,canonical_registry=registry); view=built.view; pschema=build_typed_planner_response_schema(view,require_signature_id=True); prompt=build_typed_derivation_planner_prompt(view); text,out=model_call(gen,f"{split.upper()}_PLANNER_{arm}",sid,prompt,512,schema=pschema); comp=compile_active_planner_payload(text,view,matrix)
                if not comp.ok: raise ValueError("planner_invalid:"+"|".join(comp.errors))
                closure=close_compiled_payload(comp,graph,matrix); bundle=serialize_plan_closure(closure,context=ArtifactContext(sid,rt["table_id"],arm,(role or {}).get("role_id","SCHEMA_ONLY")),initial_answer=b0m[sid]["b0_answer"]); grounds.extend(bundle.raw_groundings); derivs.extend(bundle.raw_derivations); regs.extend(bundle.registry_entries)
                state={"sample_id":sid,"arm":arm,"graph_sha256":canonical_json_hash(graph.to_dict()),"catalog_sha256":catalog.get("catalog_sha256"),"cards_sha256":canonical_json_hash(cards),"planner_view_sha256":canonical_json_hash(view),"normalized_payload":comp.normalized_payload,"allowed_signature_ids":list(comp.allowed_signature_ids),"role":role,"retrieval":retrieval,"closure_sha256":canonical_json_hash(closure.to_dict()),"planner_endpoint_record":jl(OUT/"logs/ENDPOINT_LEDGER.jsonl")[-1],"planner_usage":out.get("api_usage",{})}; w(d/"STATE"/f"{sid}_{arm}.json",state)
            except ValueError as exc: fails.append({"schema_version":"certa_active_v1_row_failure_v1","sample_id":sid,"table_id":rt["table_id"],"arm":arm,"failure_stage":"CONSTRUCTOR","error_code":str(exc).split(":",1)[0],"exception_class":"ValueError","message_sha256":hashlib.sha256(str(exc).encode()).hexdigest(),"row_preserved":True,"fallback_arm":"NONE","created_at":now()})
            ids.append(identity(rt,b0m[sid],arm,rolehash,graph_count,card_count)); seen.add((sid,arm)); wl(d/p["ids"],ids); wl(d/p["ground"],grounds); wl(d/p["deriv"],derivs); wl(d/p["reg"],regs); wl(d/p["fail"],fails)
    for rows,name in ((ids,"CONSTRUCTOR_SAMPLE_IDENTITY_SCHEMA.json"),(grounds,"RAW_GROUNDING_RECORD_SCHEMA.json"),(derivs,"RAW_DERIVATION_RECORD_SCHEMA.json"),(regs,"REGISTRY_ENTRY_SCHEMA.json")): validate_rows(rows,name)
    constructor_cost=cost_ledger((("DEV" if split=="dev" else "HOLDOUT")+"_ROLE_V3",("DEV" if split=="dev" else "HOLDOUT")+"_PLANNER_")); w(d/("CONSTRUCTOR_COST_LEDGER.json" if split=="dev" else "HOLDOUT_CONSTRUCTOR_COST_LEDGER.json"),constructor_cost)
    if split=="dev" and limit==16:
        first={x["id"] for x in runtime}; integ=OUT/"integration"; states=[j(x) for x in sorted((d/"STATE").glob("*.json")) if j(x).get("sample_id") in first]; mapping=((p["ids"],"INTEGRATION16_IDENTITIES.jsonl",ids),(p["roles"],"INTEGRATION16_ROLE_RECORDS.jsonl",roles),(p["ground"],"INTEGRATION16_RAW_GROUNDINGS.jsonl",grounds),(p["deriv"],"INTEGRATION16_RAW_DERIVATIONS.jsonl",derivs),(p["reg"],"INTEGRATION16_REGISTRY.jsonl",regs),(p["fail"],"INTEGRATION16_ROW_FAILURES.jsonl",fails),(p["pre"],"INTEGRATION16_ACTIVE_STRUCTURE_PREFLIGHT.jsonl",prefs),('',"INTEGRATION16_STATES.jsonl",states))
        for _,name,rows in mapping: wl(integ/name,[x for x in rows if x.get("sample_id") in first])
        w(integ/"INTEGRATION16_COST_LEDGER.json",constructor_cost); w(integ/"INTEGRATION_REPAIR_RECORD.json",{"repair_attempt_count":0,"repair_performed":False,"created_at":now()}); return run_tool(BASE/"tools/compute_certa_active_operational_sentinel.py","--identities",integ/"INTEGRATION16_IDENTITIES.jsonl","--derivations",integ/"INTEGRATION16_RAW_DERIVATIONS.jsonl","--repair-attempt-count","0","--output",integ/"INTEGRATION16_GATE.json")
    if split=="dev" and limit==64:
        rc=run_tool(BASE/"tools/compute_certa_active_constructor_gate.py","--identities",d/p["ids"],"--role-records",d/p["roles"],"--groundings",d/p["ground"],"--derivations",d/p["deriv"],"--registry",d/p["reg"],"--cost-ledger",d/"CONSTRUCTOR_COST_LEDGER.json","--output",d/"CONSTRUCTOR_GATE_C.json")
        if not rc: w(OUT/"freeze/CONSTRUCTOR_FREEZE.json",{"schema_version":"certa_active_constructor_freeze_v2","method_sha":git("rev-parse","HEAD"),"identities_sha256":sha(d/p["ids"]),"groundings_sha256":sha(d/p["ground"]),"derivations_sha256":sha(d/p["deriv"]),"registry_sha256":sha(d/p["reg"]),"constructor_gate_sha256":sha(d/"CONSTRUCTOR_GATE_C.json"),"created_at":now()})
        return rc
    return 0

def rebuild_closure(rt, state, matrix):
    table=load_table_for_cscr(rt,str(TABLES),{},"hitab"); graph=build_hceg(table,rt["question"]); payload=state["normalized_payload"]; comp=ActiveCompilationResult(True,payload,canonical_json(payload),canonical_json_hash(payload),tuple(state["allowed_signature_ids"]),()); return graph,close_compiled_payload(comp,graph,matrix)

def decision(split):
    p=paths(split); d=p["dir"]; runtime=jl(p["runtime"]); b0m={x["sample_id"]:x for x in jl(p["b0"])}; rolem={x["sample_id"]:x for x in jl(d/p["roles"])}; raw=jl(d/p["deriv"]); regs=jl(d/p["reg"]); cm=j(OUT/"freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json"); dm=j(OUT/"freeze/DECISION_CAPABILITY_MATRIX.json"); active={x["role_id"] for x in dm["rows"] if x["decision_active"]}; contexts=[]; vault={}
    for rt in runtime:
        sid=rt["id"]; statep=d/"STATE"/f"{sid}_C2_ROLE_RETRIEVAL.json"; role=rolem[sid].get("canonical_record"); graph=closure=contrast=None
        if statep.is_file():
            state=j(statep); graph,closure=rebuild_closure(rt,state,cm); part=partition_support(closure,initial_proposal_answer=b0m[sid]["b0_answer"]); basis=build_sample_fixed_role_intervention_basis(closure.executable_derivations,graph); classes=build_basis_relative_behavior_classes(closure.executable_derivations,graph,basis); contrast=build_compact_behavioral_contrast_v3(derivations=closure.executable_derivations,behavior_classes=classes,basis=basis,original_answer=b0m[sid]["b0_answer"],query_semantics=role_v3_to_planner_query_contract(role) if role else {})
        else: part=None
        executed=tuple(closure.executable_derivations) if closure else (); elig=assess_decision_eligibility(role_id=(role or {}).get("role_id","UNSUPPORTED"),decision_active_role_ids=active,support_partition=part,compact_contrast=contrast or {},executed_derivations=executed); packet={"query_contract":role_v3_to_planner_query_contract(role) if role and role.get("supported") else {},"compact_behavioral_contrast_v3":contrast.to_dict() if contrast else {},"metadata":{"sample_id":sid,"role_record_sha256":rolem[sid].get("record_sha256","")}}; prompt=build_cera_prompt(packet,template_version=CERA_V3_TEMPLATE_VERSION) if elig.eligible else ""; contexts.append((rt,role,executed,contrast,elig,packet,prompt))
        for x in executed:
            h=active_answer_hash(x.projected_answer)
            if h in vault and vault[h]!=x.projected_answer: raise RuntimeError("answer_vault_hash_collision")
            vault[h]=x.projected_answer
    artifacts=decision_artifact_paths(split); prefix=artifacts["prefix"]; dd=artifacts["dir"]; eligibility={"schema_version":"certa_active_v1_decision_eligibility_v1","eligible_sample_ids":[x[0]["id"] for x in contexts if x[4].eligible],"rows":[{"sample_id":x[0]["id"],"role_id":x[1].get("role_id") if x[1] else "UNSUPPORTED","eligible":x[4].eligible,"cera_call_allowed":x[4].cera_call_allowed,"failure_reasons":list(x[4].failure_reasons),"prompt_sha256":hashlib.sha256(x[6].encode()).hexdigest() if x[6] else ""} for x in contexts]}; w(artifacts["eligibility"],eligibility); wl(dd/f"{prefix}_FROZEN_ANSWER_VAULT.blind.jsonl",[{"answer_hash":k,"executed_answer":v} for k,v in sorted(vault.items())]); wl(dd/f"{prefix}_FROZEN_CONTRAST_PROVENANCE.blind.jsonl",[{"sample_id":x[0]["id"],"contrast":x[3].to_dict()} for x in contexts if x[3]])
    gen=generator(); templates=[{"sample_id":x[0]["id"],"prompt":x[6],"prompt_sha256":hashlib.sha256(x[6].encode()).hexdigest(),"request_sha256":canonical_json_hash(gen._completion_request_kwargs(prompt=x[6],max_new_tokens=512,temperature=0,top_p=1))} for x in contexts if x[6]]; wl(dd/f"{prefix}_CERA_REQUEST_TEMPLATES.blind.jsonl",templates); schema_prompt=templates[0]["prompt"] if templates else build_cera_prompt({"query_contract":{},"compact_behavioral_contrast_v3":{},"metadata":{}},template_version=CERA_V3_TEMPLATE_VERSION)
    if split=="dev": w(OUT/"freeze/PRIMARY_DECISION_FREEZE.json",{"schema_version":"certa_active_primary_decision_freeze_v2","primary_active_arm":"CERA_PLUS_VALIDATOR","control_arm":"DETERMINISTIC_SELECTOR","fallback":"B0_KEEP","eligible_only_cera":True,"prompt_sha256":canonical_json_hash(templates),"response_schema_sha256":canonical_json_hash(cera_response_contract(schema_prompt)),"validator_sha256":sha(REPO/"certa/repair/safety_validator.py"),"materializer_sha256":sha(REPO/"certa/active_v1/decision_adapter.py"),"registry_sha256":sha(d/p["reg"]),"model_profile_sha256":sha(PROFILE),"created_at":now(),"method_sha":git("rev-parse","HEAD")}); validate_rows([j(OUT/"freeze/PRIMARY_DECISION_FREEZE.json")],"PRIMARY_DECISION_FREEZE_SCHEMA.json")
    elif sha(REPO/"certa/active_v1/decision_adapter.py")!=j(OUT/"freeze/PRIMARY_DECISION_FREEZE.json")["materializer_sha256"]: raise RuntimeError("holdout_method_drift")
    decisions=[]; validators=[]; finals=[]; reconciliations=[]; controls=[]
    for idx,(rt,role,executed,contrast,elig,packet,prompt) in enumerate(contexts):
        sid=rt["id"]; sample_raw=[x for x in raw if x["sample_id"]==sid and x["arm"]=="C2_ROLE_RETRIEVAL"]; sample_regs=[x for x in regs if x["sample_id"]==sid and x["arm"]=="C2_ROLE_RETRIEVAL"]; output=validator=None
        if elig.eligible: text,_=model_call(gen,f"{prefix}_CERA",sid,prompt,512); output=text; validator=validate_cera_output_v3(text,packet)
        created=now(); resolution=reconcile_cera_decision(eligibility=elig,raw_output=output,validator=validator,compact_contrast=contrast or {},executed_derivations=executed,raw_derivation_records=sample_raw,registry_entries=sample_regs,b0_answer=b0m[sid]["b0_answer"],sample_id=sid,decision_id=f"{prefix}-DEC-{idx:03d}",validator_record_id=f"{prefix}-VAL-{idx:03d}",created_at=created); final=materialize_selected_final(resolution,b0_answer=b0m[sid]["b0_answer"],materialized_at=now()); decisions.append(resolution.decision_record); finals.append(final.record); reconciliations.append(resolution.reconciliation_record)
        if resolution.validator_record: validators.append(resolution.validator_record)
        alt=(contrast.to_dict().get("alternative_hypothesis",{}) if contrast else {}); controls.extend([{"sample_id":sid,"decision_arm":"B0_KEEP","action":"KEEP_B0"},{"sample_id":sid,"decision_arm":"DETERMINISTIC_SELECTOR","action":"USE_UNIQUE_ALTERNATIVE" if elig.eligible else "KEEP_B0","selected_hypothesis_id":alt.get("hypothesis_id")}])
    for rows,name in ((decisions,"DECISION_RECORD_SCHEMA.json"),(validators,"VALIDATOR_RECORD_SCHEMA.json"),(finals,"SELECTED_FINAL_RECORD_SCHEMA.json"),(reconciliations,"REGISTRY_SELECTED_FINAL_RECONCILIATION_SCHEMA.json")): validate_rows(rows,name)
    wl(dd/f"{prefix}_DECISIONS.blind.jsonl",decisions); wl(dd/f"{prefix}_VALIDATOR_RECORDS.blind.jsonl",validators); wl(dd/f"{prefix}_SELECTED_FINALS.blind.jsonl",finals); wl(dd/f"{prefix}_RECONCILIATION.blind.jsonl",reconciliations); wl(dd/f"{prefix}_CONTROL_DECISIONS.blind.jsonl",controls); w(dd/f"{prefix}_DECISION_COST.json",cost_ledger((prefix+"_CERA",))); close={"schema_version":"certa_active_v1_selected_final_prediction_close_v1","split":split,"method_sha":git("rev-parse","HEAD"),"eligibility_sha256":sha(artifacts["eligibility"]),"decisions_sha256":sha(dd/f"{prefix}_DECISIONS.blind.jsonl"),"selected_finals_sha256":sha(dd/f"{prefix}_SELECTED_FINALS.blind.jsonl"),"logical_cera_calls":len(eligibility["eligible_sample_ids"]),"closed_at":now()}; w(artifacts["close"],close); return 0 if eligibility["eligible_sample_ids"] else 2

def unblind(split):
    p=paths(split); d=p["dir"]; artifacts=decision_artifact_paths(split); prefix=artifacts["prefix"]; dd=artifacts["dir"]; close=artifacts["close"]
    if not close.is_file(): raise RuntimeError("prediction_close_missing")
    source=DEV_SOURCE if split=="dev" else HOLD_SOURCE; source_rows=jl(source); runtime=jl(p["runtime"]); sm={str(x.get("id")):x for x in source_rows}; gold=[]
    for rt in runtime:
        row=sm[rt["id"]]; answer=row.get("answer",row.get("answers",row.get("gold_answer",""))); gold.append({"schema_version":"certa_active_gold_record_v2","fixture_only":False,"sample_id":rt["id"],"table_id":rt["table_id"],"gold_answer_hash":active_answer_hash(answer)})
    up=OUT/("unblind" if split=="dev" else "holdout"); gp=up/f"{prefix}_GOLD.jsonl"; validate_rows(gold,"GOLD_RECORD_SCHEMA.json"); wl(gp,gold); access={"schema_version":"certa_active_access_log_v2","resource":str(source),"sha256":sha(gp),"accessed_at":now(),"accessor":"independent_post_close_analyzer","purpose":f"single_{split}_unblind"}; ap=up/f"{prefix}_GOLD_ACCESS_LOG.json"; validate_rows([access],"ACCESS_LOG_SCHEMA.json"); w(ap,access)
    ids=d/p["ids"]; decisions=dd/f"{prefix}_DECISIONS.blind.jsonl"; finals=dd/f"{prefix}_SELECTED_FINALS.blind.jsonl"; validators=dd/f"{prefix}_VALIDATOR_RECORDS.blind.jsonl"; recout=up/f"{prefix}_RECONCILIATION.jsonl"; trans=up/f"{prefix}_TRANSITIONS.jsonl"; gate=up/("DEV_DECISION_GATE.json" if split=="dev" else "HOLDOUT_GATE.json")
    if split=="dev":
        rc1=run_tool(BASE/"tools/compute_certa_active_opportunity_gate.py","--identities",ids,"--derivations",d/p["deriv"],"--registry",d/p["reg"],"--gold",gp,"--constructor-freeze",OUT/"freeze/CONSTRUCTOR_FREEZE.json","--gold-access-log",ap,"--output",up/"OPPORTUNITY_GATE_O.json")
    else: rc1=0
    rc2=run_tool(BASE/"tools/compute_certa_active_decision_gate.py","--identities",ids,"--decisions",decisions,"--selected-finals",finals,"--validators",validators,"--registry",d/p["reg"],"--derivations",d/p["deriv"],"--gold",gp,"--primary-freeze",OUT/"freeze/PRIMARY_DECISION_FREEZE.json","--gold-access-log",ap,"--cost-ledger",dd/f"{prefix}_DECISION_COST.json","--reconciliation-out",recout,"--transitions-out",trans,"--output",gate,"--split",split)
    if split=="dev":
        rm={x["sample_id"]:x for x in jl(d/p["roles"])}; stats=[{"sample_id":x["sample_id"],"table_id":next(i["table_id"] for i in jl(ids) if i["sample_id"]==x["sample_id"]),"b0_correct":x["b0_correct"],"selected_correct":x["selected_correct"],"selected_changed":x["b0_answer_hash"]!=x["selected_final_answer_hash"],"operation_family":(rm[x["sample_id"]].get("canonical_record") or {}).get("operation_family","UNSUPPORTED")} for x in jl(trans)]; sp=up/"SELECTED_FINAL_STATISTICS_INPUT.jsonl"; wl(sp,stats); run_tool(PACK/"tools/compute_selected_final_statistics.py","--records",sp,"--output",up/"SELECTED_FINAL_STATISTICS.json")
    return rc1 or rc2

def preflight():
    current=OUT/"intake/PYTHON_RUNTIME_PREFLIGHT.json"; prior=OUT/"intake/PYTHON_RUNTIME_PREFLIGHT_PRE_IMPLEMENTATION.json"
    if current.is_file() and not prior.exists(): shutil.copyfile(current,prior)
    return run_tool(BASE/"runtime_preflight.py","--repo",REPO,"--embedding-path","/home/common_data/llm/intfloat/multilingual-e5-large","--models-url","http://127.0.0.1:30338/v1/models","--expected-model","Qwen3-8B","--output",current)

def holdout_blind():
    if not j(OUT/"unblind/OPPORTUNITY_GATE_O.json").get("pass") or not j(OUT/"unblind/DEV_DECISION_GATE.json").get("pass"): raise RuntimeError("holdout_not_authorized")
    w(OUT/"holdout/HOLDOUT_METHOD_FREEZE.json",{"schema_version":"certa_active_v1_holdout_method_freeze_v1","method_sha":git("rev-parse","HEAD"),"implementation_freeze_sha256":sha(OUT/"freeze/IMPLEMENTATION_SOURCE_FREEZE.json"),"constructor_freeze_sha256":sha(OUT/"freeze/CONSTRUCTOR_FREEZE.json"),"primary_decision_freeze_sha256":sha(OUT/"freeze/PRIMARY_DECISION_FREEZE.json"),"created_at":now()}); ensure_holdout_b0(); rc=constructor("holdout",64,("C2_ROLE_RETRIEVAL",)); return rc or decision("holdout")

def finalize(state):
    if git("status","--porcelain"): raise RuntimeError("finalize_requires_clean_worktree")
    td=OUT/"terminal"; w(td/"COST_LEDGER.json",cost_ledger()); required=[x.strip() for x in (PACK/"REQUIRED_ARTIFACTS.md").read_text().splitlines() if "/" in x and not x.startswith("#") and not x.startswith("```")]; disposition=[]
    for item in required:
        for path in item.split():
            if "/" not in path: continue
            p=OUT/path.strip("`;,"); disposition.append({"path":str(p.relative_to(OUT)),"status":"PRESENT" if p.is_file() else "NOT_REACHED","sha256":sha(p) if p.is_file() else None})
    w(td/"REQUIRED_ARTIFACTS.json",{"schema_version":"certa_active_v1_required_artifacts_v1","terminal_state":state,"artifacts":disposition}); w(td/"FINAL_METHOD_FREEZE_MANIFEST.json",{"schema_version":"certa_active_v1_final_method_freeze_v1","terminal_state":state,"method_sha":git("rev-parse","HEAD"),"implementation_freeze_sha256":sha(OUT/"freeze/IMPLEMENTATION_SOURCE_FREEZE.json"),"worktree_clean":True,"created_at":now()}); w(td/"FINAL_TERMINAL_STATE.json",{"schema_version":"certa_active_v1_final_completion_terminal_v1","terminal_state":state,"method_sha":git("rev-parse","HEAD"),"method_frozen":True,"no_new_method_round":True,"experiment_only_authorized":state=="FREEZE_CERTA_ACTIVE_METHOD_EXPERIMENT_ONLY","created_at":now()})
    bundle=td/"CERTA_ACTIVE_V1_FINAL_COMPLETION.bundle"; subprocess.run(["git","bundle","create",str(bundle),"research/certa-active-v1"],cwd=REPO,check=True); subprocess.run(["git","bundle","verify",str(bundle)],cwd=REPO,check=True,capture_output=True); files=sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="SHA256SUMS.txt"); (td/"SHA256SUMS.txt").write_text("".join(f"{sha(x)}  {x.relative_to(OUT)}\n" for x in files)); (td/"TERMINAL_REPORT.md").write_text(f"# CERTA Active V1 Final Completion\n\nTerminal: `{state}`\n\nCommit: `{git('rev-parse','HEAD')}`\n\nLogical calls: `{cost_ledger()['logical_calls']}`\n",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("freeze","preflight","integration16","constructor64","decision-dev","unblind-dev","holdout-blind","unblind-holdout","finalize")); ap.add_argument("--state"); a=ap.parse_args(); rc=0
    if a.command=="freeze": freeze()
    elif a.command=="preflight": rc=preflight()
    elif a.command=="integration16": rc=constructor("dev",16,ARMS)
    elif a.command=="constructor64": rc=constructor("dev",64,ARMS)
    elif a.command=="decision-dev": rc=decision("dev")
    elif a.command=="unblind-dev": rc=unblind("dev")
    elif a.command=="holdout-blind": rc=holdout_blind()
    elif a.command=="unblind-holdout": rc=unblind("holdout")
    elif a.command=="finalize":
        if not a.state: raise SystemExit("--state required")
        finalize(a.state)
    raise SystemExit(rc)

if __name__=="__main__": main()
