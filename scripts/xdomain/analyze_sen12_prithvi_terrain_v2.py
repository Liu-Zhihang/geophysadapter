#!/usr/bin/env python3
"""Strict paired analysis for Prithvi-EO-2.0 and Terrain-v2 Sen12 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import analyze_sen12_xdomain_geophysadapter as stats


ROOT = Path(__file__).resolve().parents[2]
RUN_PATTERN = re.compile(r"fold(?P<fold>\d+)_seed(?P<seed>\d+)$")
ARTIFACTS = ("DONE.json", "result.json", "checkpoint.pt", "config.json", "per_sample.csv", "per_event.csv", "per_region.csv")
DATA_FIELDS = (
    "fold", "seed", "prithvi_checkpoint_sha256", "base_h5", "optical_h5",
    "terrain_h5", "split_csv_sha256", "sample_sha256",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "experiments/revision2026/sen12_prithvi_terrain_v2_pilot")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--expected-folds", default="0,1,2,3,4")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def actual_signature(recorded: dict, label: str, run_dir: Path) -> None:
    path = Path(str(recorded.get("path", ""))).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{label} missing for {run_dir}: {path}")
    actual = {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    normalized = {"path": str(Path(str(recorded.get("path"))).resolve()), "size": recorded.get("size"), "mtime_ns": recorded.get("mtime_ns")}
    if actual != normalized:
        raise RuntimeError(f"{label} signature mismatch for {run_dir}")


def load_run(run_dir: Path, mode: str, fold: int, seed: int):
    for name in ARTIFACTS:
        stats.require_nonempty_file(run_dir / name)
    done, result, config = (load_json(run_dir / name) for name in ("DONE.json", "result.json", "config.json"))
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    for label, payload in (("DONE", done), ("result", result.get("identity", {})), ("config", config), ("checkpoint", checkpoint.get("identity", {}))):
        expected = {"mode": mode, "fold": fold, "seed": seed}
        mismatch = {key:(value,payload.get(key)) for key,value in expected.items() if payload.get(key)!=value}
        if mismatch: raise RuntimeError(f"{label} identity mismatch in {run_dir}: {mismatch}")
    if done.get("status") != "complete": raise RuntimeError(f"incomplete run: {run_dir}")
    if done.get("result_sha256") != stats.sha256_file(run_dir / "result.json"): raise RuntimeError(f"result hash mismatch: {run_dir}")
    if done.get("checkpoint_sha256") != stats.sha256_file(run_dir / "checkpoint.pt"): raise RuntimeError(f"checkpoint hash mismatch: {run_dir}")
    identity = result.get("identity")
    if identity != checkpoint.get("identity") or identity != config.get("identity"):
        raise RuntimeError(f"result/checkpoint/config identity differs: {run_dir}")
    if checkpoint.get("trainable_sha256") != result.get("trainable_sha256") or checkpoint.get("trainable_sha256") != done.get("trainable_sha256"):
        raise RuntimeError(f"trainable state hash mismatch: {run_dir}")
    if tensor_dict_sha256(checkpoint["trainable_state_dict"]) != checkpoint.get("trainable_sha256"):
        raise RuntimeError(f"checkpoint trainable tensor content hash mismatch: {run_dir}")
    for field in ("base_h5", "optical_h5", "terrain_h5"):
        actual_signature(identity[field], field, run_dir)
    split_path = Path(config["args"]["split_csv"])
    if stats.sha256_file(split_path) != identity["split_csv_sha256"]:
        raise RuntimeError(f"split CSV hash mismatch: {run_dir}")
    return {"done":done,"result":result,"config":config,"checkpoint":checkpoint,"identity":identity}


def data_identity(identity: dict) -> dict:
    return {key:identity.get(key) for key in DATA_FIELDS}


def load_pair(pair_dir: Path, fold: int, seed: int):
    visual = load_run(pair_dir / "visual", "visual", fold, seed)
    adapter = load_run(pair_dir / "adapter", "adapter", fold, seed)
    if data_identity(visual["identity"]) != data_identity(adapter["identity"]):
        raise RuntimeError(f"visual/adapter data identity differs: {pair_dir}")
    expected_parent = str((pair_dir / "visual/checkpoint.pt").resolve())
    if adapter["identity"].get("parent_visual_checkpoint") != expected_parent:
        raise RuntimeError(f"adapter parent path mismatch: {pair_dir}")
    if adapter["identity"].get("parent_visual_trainable_sha256") != visual["checkpoint"].get("trainable_sha256"):
        raise RuntimeError(f"adapter parent trainable hash mismatch: {pair_dir}")
    result = adapter["result"]
    if result.get("frozen_visual_sha256_before") != result.get("frozen_visual_sha256_after"):
        raise RuntimeError(f"frozen visual decoder changed: {pair_dir}")
    if adapter["checkpoint"].get("threshold_source") != "loaded_matched_visual_checkpoint":
        raise RuntimeError(f"adapter threshold source mismatch: {pair_dir}")
    if float(adapter["checkpoint"]["threshold"]) != float(visual["checkpoint"]["threshold"]):
        raise RuntimeError(f"visual/adapter threshold mismatch: {pair_dir}")
    audits = [item for item in result.get("identity_and_control_audits", []) if item.get("split")=="test"]
    if len(audits)!=1: raise RuntimeError(f"missing test audit: {pair_dir}")
    audit=audits[0]
    if not (audit.get("same_sample_identity_and_order") and audit.get("zero_terrain_exact_fallback") and audit.get("q_t_zero_exact_fallback")):
        raise RuntimeError(f"fallback or sample audit failed: {pair_dir}")
    if int(audit.get("other_region_donor_violations",-1)) != 0: raise RuntimeError(f"donor violation: {pair_dir}")

    outputs={}
    for filename,unit in (("per_sample.csv","sample_id"),("per_event.csv","physical_event_id"),("per_region.csv","region_group")):
        required=(unit,"mode","fold","seed","split","control","iou","average_precision","brier")
        vf=stats.assert_row_contract(stats.load_nonempty_csv(pair_dir/"visual"/filename,required),pair_dir/"visual"/filename,"visual",fold,seed)
        af=stats.assert_row_contract(stats.load_nonempty_csv(pair_dir/"adapter"/filename,required),pair_dir/"adapter"/filename,"adapter",fold,seed)
        stats.assert_control_coverage(vf,unit,("visual",),pair_dir/"visual"/filename)
        stats.assert_control_coverage(af,unit,stats.ADAPTER_CONTROLS,pair_dir/"adapter"/filename)
        stats.assert_visual_anchor_equal(vf,af,unit,fold,seed)
        outputs[filename]=af
    corpus=stats.corpus_test_frame(adapter["result"],fold,seed)
    return outputs,corpus


def main() -> int:
    args=parse_args(); expected={int(x) for x in args.expected_folds.split(",") if x.strip()}
    pairs={}
    for path in sorted(args.runs_dir.iterdir()):
        match=RUN_PATTERN.fullmatch(path.name) if path.is_dir() else None
        if match: pairs.setdefault(int(match["fold"]),{})[int(match["seed"])]=path
    if not pairs: raise RuntimeError(f"no pairs under {args.runs_dir}")
    if not args.allow_partial:
        if set(pairs)!=expected: raise RuntimeError(f"fold coverage mismatch: {sorted(pairs)}")
        if any(len(pairs[fold])<args.min_seeds for fold in expected): raise RuntimeError("insufficient seeds")
    frames={"sample":[],"event":[],"region":[],"run":[]}
    artifact_rows=[]
    for fold in sorted(pairs):
        for seed,path in sorted(pairs[fold].items()):
            outputs,corpus=load_pair(path,fold,seed)
            frames["sample"].append(outputs["per_sample.csv"])
            frames["event"].append(outputs["per_event.csv"])
            frames["region"].append(outputs["per_region.csv"])
            frames["run"].append(corpus)
            artifact_rows.append({"fold":fold,"seed":seed,"status":"PASS","pair_dir":str(path.resolve())})
    combined={key:pd.concat(value,ignore_index=True) for key,value in frames.items()}
    paired={
        "run":stats.paired_table(combined["run"],()),
        "sample":stats.paired_table(combined["sample"],("sample_id",)),
        "event":stats.paired_table(combined["event"],("physical_event_id",)),
        "region":stats.paired_table(combined["region"],("region_group",)),
    }
    min_seeds=min(len(value) for value in pairs.values())
    summary={
        "status":"DEVELOPMENT_PARTIAL" if args.allow_partial else "PASS",
        "n_paired_runs":sum(len(value) for value in pairs.values()),
        "folds":sorted(pairs),"observed_min_seeds_per_fold":min_seeds,"required_min_seeds":args.min_seeds,
        "control_summary":stats.control_summary(combined["run"]),
        "statistics":{
            "run_corpus_fold_bootstrap":stats.summarize_paired(paired["run"],(),("fold",),args.bootstrap,args.permutations),
            "physical_event_bootstrap":stats.summarize_paired(paired["event"],("physical_event_id",),("physical_event_id",),args.bootstrap,args.permutations),
            "region_bootstrap":stats.summarize_paired(paired["region"],("region_group",),("region_group",),args.bootstrap,args.permutations),
        },
    }
    out=args.outdir or args.runs_dir/"analysis_partial"
    out.mkdir(parents=True,exist_ok=True)
    for key,name in (("run","per_run_paired.csv"),("sample","per_sample_paired.csv"),("event","per_event_paired.csv"),("region","per_region_paired.csv")):
        stats.clean_csv(paired[key],out/name)
    (out/"summary.json").write_text(json.dumps(stats.json_safe(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"report.md").write_text(stats.build_report(summary),encoding="utf-8")
    (out/"artifact_check.json").write_text(json.dumps({"status":summary["status"],"pairs":artifact_rows},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":summary["status"],"pairs":summary["n_paired_runs"],"outdir":str(out)},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
