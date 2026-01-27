from __future__ import annotations
# src/cli.py

import os
# If you want deterministic cuBLAS, set it here (or via .env) BEFORE importing torch/cuda:
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Import torch only if this file actually uses it; otherwise remove this line.
# import torch

# ---- tracing helpers (top of src/cli.py) ----
import time as _time

def _t():
    return _time.monotonic()

def _ms(dt):
    return f"{dt*1000:.1f} ms"

def _p(msg: str):
    print(msg, flush=True)
# ---------------------------------------------

from pathlib import Path
import json
import random
import time
import traceback
from typing import Optional, List, Dict, Any

import typer
from rich import print
from rich.table import Table
from dotenv import load_dotenv

from src.config import load_app_config
from src.data.loader import (
    build_all,
    jbb_category_counts,
    validate_m5_ready,
    load_seeds_for_m5,
    load_operators_for_m5,
    # NEW: OOD exporters
    export_jbb_harmful_to_seeds_ood,
    
)
from src.store.run_store import RunStore
from src.models.router import RouterClient
from src.models.adapters import make_llm, TargetAdapter
from src.rewriter.rewriter_llm import PromptRewriter, RewriterInput
from src.judge.judge_llm import Judge, JudgeInput
from src.reward.rewarder import RewardCalculator, RewardInput
from src.data.loader import _load_seeds_jsonl  # canonical

app = typer.Typer(add_completion=False)

def _env():
    load_dotenv(override=False)

# ----------------------------
# Helpers
# ----------------------------
def _resolve_target_model_id(cfg: dict, override: Optional[str] = None) -> Optional[str]:
    """Supports either dict or string entries in config.targets."""
    if override:
        return override
    targets = cfg.get("targets", [])
    if not targets:
        return None
    first = targets[0]
    return first.get("id") if isinstance(first, dict) else str(first)



# ----------------------------
# M1: bootstrap & config
# ----------------------------
@app.command(help="Create/verify folders from config and print summary.")
def bootstrap(
    config_path: str = typer.Option("config/stack.yaml", "--config-path", help="Path to unified config."),
):
    _env()
    cfg = load_app_config(config_path)
    paths = cfg.get("paths", {})
    for key in ("data", "artifacts", "runs"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    print("M1: bootstrap complete")
    print(f"  data      -> {paths['data']}")
    print(f"  artifacts -> {paths['artifacts']}")
    print(f"  runs      -> {paths['runs']}")
    print(f"  router    -> {cfg.get('router', {}).get('base_url')}")

@app.command(help="Show merged app config (env-expanded).")
def show_config(
    config_path: str = typer.Option("config/stack.yaml", "--config-path", help="Path to unified config."),
):
    _env()
    cfg = load_app_config(config_path)
    print(json.dumps(cfg, indent=2))

# ----------------------------
# router health
# ----------------------------
@app.command(help="List models on the router and show health.")
def router_health(
    config_path: str = typer.Option("config/stack.yaml", "--config-path", help="Path to unified config."),
):
    _env()
    cfg = load_app_config(config_path)
    rc = RouterClient(base_url=cfg["router"]["base_url"])
    print(f"[router_health] GET {rc.base_url}/v1/models")
    data = rc.list_models()
    models = data.get("data") or data.get("models") or []
    table = Table("index", "id")
    for i, m in enumerate(models):
        mid = m.get("id") or m.get("model") or m
        table.add_row(str(i), str(mid))
    print(f"Router models @ {rc.base_url} ")
    print(table)

# ----------------------------
# echo (single, adapter-based)
# ----------------------------
@app.command(help="Simple echo call to a model (via adapters).")
def echo(
    model: str = typer.Option(..., "--model", help="Model id, e.g. ollama/llama3:instruct"),
    text: str = typer.Option(..., "--text", help="Prompt text to send"),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    backend: str = typer.Option("router", "--backend", help='"router" or "ollama_direct"'),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose debug logging."),
):
    _env()
    cfg = load_app_config(config_path)
    router_base = cfg.get("router", {}).get("base_url")

    # Only require router when NOT using an Ollama model id
    if (not model.startswith("ollama/")) and not router_base and backend != "ollama_direct":
        raise RuntimeError("[config] router.base_url is missing. Set it in config/stack.yaml or export ROUTER_BASE.")

    if verbose:
        print(f"[echo] router_base={router_base} model={model} backend={backend}")

    t0 = time.monotonic()
    llm = make_llm(
        model_id=model,
        router_base=router_base,
        backend=("ollama_direct" if backend == "ollama_direct" else None),
    )
    target = TargetAdapter(llm)

    if verbose:
        print("[echo] calling TargetAdapter.generate()...")
    t1 = time.monotonic()
    resp = target.generate(text, temperature=temperature, max_tokens=max_tokens)
    t2 = time.monotonic()

    if verbose:
        print(f"[echo] construct_llm_ms={(t1 - t0)*1000:.1f} call_ms={(t2 - t1)*1000:.1f}")

    print(f"model: {resp.get('model')}")
    print(f"latency_ms: {resp.get('latency_ms')}")
    print(f"text: {resp.get('text','').strip()}")

# ----------------------------
# M2: run-init
# ----------------------------
@app.command(help="Initialize a per-model run folder and snapshot data.")
def run_init(
    model: str = typer.Option(..., "--model", help="Model id for naming the run folder."),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
):
    _env()
    cfg = load_app_config(config_path)
    paths = cfg["paths"]
    data_root = Path(paths["data"])
    runs_root = Path(paths["runs"])
    redact_in_csv = cfg.get("logging", {}).get("redact_in_csv", False)

    run = RunStore(runs_root=runs_root, model_name=model, redact_in_csv=redact_in_csv)
    stats = build_all(data_root=data_root, run_dir=run.root)

    jbb_stats = jbb_category_counts(data_root / "jbb")
    run.write_meta(data_root=data_root, jbb_stats=jbb_stats)

    print("[M2] run initialized")
    print(f"  run_dir        : {run.root}")
    print(f"  seeds.jsonl    : {stats['seeds_rows']} rows")
    print(f"  operators.json : {stats['operators']} operators")
    print(f"  categories.json: {stats['categories']} categories")
    pq = stats['pacing']['turns_quantiles']
    print(f"  pacing.json    : T_default={stats['pacing']['T_default']}  q25/q50/q75={pq['q25']}/{pq['q50']}/{pq['q75']}")
    run.close()
    print("Run initialization complete.")

# ----------------------------
# NEW: OOD export
# ----------------------------
@app.command("export-ood", help="Export OOD seeds (JBB harmful) to the current run.")
def export_ood(
    n: int = typer.Option(100, "--n", help="Max rows to export (uses first n rows)."),
    runs_model: Optional[str] = typer.Option(None, "--runs-model"),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
):
    _env()
    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])
    data_root = Path(cfg["paths"]["data"])

    model_for_run = runs_model or _resolve_target_model_id(cfg, None) or "unspecified"
    rs = RunStore(runs_root=runs_root, model_name=model_for_run)

    from src.data.loader import export_jbb_harmful_to_seeds_ood
    out_jbb = Path(rs.root) / "seeds_ood.jbb.jsonl"
    total = export_jbb_harmful_to_seeds_ood(data_root / "jbb", out_jbb)
    print(f" jbb: wrote {total} rows -> {out_jbb}")


# ----------------------------
# M5: rewrite-sample (supports OOD override)
# ----------------------------
@app.command("rewrite-sample", help="M5: produce RAW rewritten prompts from SAFE or OOD seeds/operators and log events.")
def rewrite_sample(
    n: int = typer.Option(20, "--n", "-n", help="Number of rewrites to generate.", show_default=True),
    send_to_target: bool = typer.Option(False, help="Immediately send each rewritten prompt to a target model for smoke testing."),
    target_model: Optional[str] = typer.Option(None, "--target-model", help="Override target model id (defaults to first entry in config.targets)."),
    seeds_path_override: Optional[str] = typer.Option(None, "--seeds-path-override", help="Path to a JSONL seeds file (e.g., seeds_ood.jbb.jsonl)."),
    dataset_tag: Optional[str] = typer.Option(None, "--dataset-tag", help='Tag events with dataset source, e.g. "OOD-JBB", "OOD-AdvBench".'),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose debug logging."),
):
    _env()
    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])
    redact_in_csv = cfg.get("logging", {}).get("redact_in_csv", False)

    # use model name only to create/find a run dir; for M5 we can use a neutral tag
    model_for_run = target_model or _resolve_target_model_id(cfg, None) or "unspecified"
    rs = RunStore(runs_root=runs_root, model_name=model_for_run, redact_in_csv=redact_in_csv)

    # ensure M2 snapshots exist & are usable (for operators.json)
    n_seeds_m5, n_ops = validate_m5_ready(rs)
    if verbose:
        print(f"[M5] ready: seeds(snapshot)={n_seeds_m5} operators={n_ops}")

    # load operators from SORRY mutators
    ops = load_operators_for_m5(rs)

    # choose seed source
    if seeds_path_override:
        seeds = _load_seeds_jsonl(Path(seeds_path_override))
        if verbose:
            print(f"[M5] using OOD seeds from: {seeds_path_override} ({len(seeds)} rows)")
    else:
        seeds = load_seeds_for_m5(rs)
        if verbose:
            print(f"[M5] using in-distribution seeds.jsonl from run ({len(seeds)} rows)")

    rewriter = PromptRewriter(cfg, rs)
    if verbose:
        rid = cfg.get("rewriter", {}).get("model_id")
        rbackend = cfg.get("rewriter", {}).get("backend")
        print(f"[M5] rewriter model={rid} backend={rbackend}")

    tgt = None
    if send_to_target:
        mid = _resolve_target_model_id(cfg, target_model)
        if not mid:
            raise RuntimeError("No target model configured in config.targets and no --target-model provided.")
        router_base = cfg.get("router", {}).get("base_url")
        # Only require router when the target is NOT an Ollama model id
        if (not mid.startswith("ollama/")) and not router_base:
            raise RuntimeError("[config] router.base_url is missing. Set it in config/stack.yaml or export ROUTER_BASE.")
        if verbose:
            print(f"[M5] building target LLM for {mid} (router_base={router_base})")
        llm = make_llm(model_id=mid, router_base=router_base)  # auto-directs ollama/* to OllamaLLM
        tgt = TargetAdapter(llm)
        if verbose:
            print("[M5] target adapter ready")

    persist_raw = cfg.get("logging", {}).get("persist_raw_prompts", True)
    if verbose:
        print(f"[M5] persist_raw_prompts={persist_raw}")

    # Main loop
    for i in range(n):
        seed = random.choice(seeds)
        op = random.choice(ops)["name"]

        sliders = {
            "length": random.random(),
            "temperature": random.random(),
            "persona": random.random(),
        }

        if verbose:
            print(f"\n[M5][{i+1}/{n}] seed_id={seed.get('question_id')} cat={seed.get('category')} op={op}")
            print(f"[M5][{i+1}/{n}] sliders={sliders}")

        rin = RewriterInput(
            seed_id=seed["question_id"],
            category=seed.get("category", "Uncategorized"),
            seed_text=seed["text"],
            operator_name=op,
            sliders=sliders,
        )

        # --- Call rewriter with timing and error handling
        t0 = time.monotonic()
        try:
            if verbose:
                print(f"[M5][{i+1}/{n}] calling rewriter.rewrite() ...")
            r = rewriter.rewrite(rin)
            t1 = time.monotonic()
            if verbose:
                text_preview = (r.text or "").strip().replace("\n", " ")[:120]
                print(f"[M5][{i+1}/{n}] rewriter OK in {(t1 - t0)*1000:.1f} ms | len(text)={len(r.text)} | meta={r.meta}")
                print(f"[M5][{i+1}/{n}] text preview: {text_preview!r}")
        except Exception as e:
            t1 = time.monotonic()
            print(f"[M5][{i+1}/{n}] rewriter ERROR after {(t1 - t0)*1000:.1f} ms: {e}")
            if verbose:
                traceback.print_exc()
            # Still log an event with error meta, then continue
            rs.append_event({
                "phase": "rewrite_error",
                "seed_id": rin.seed_id,
                "category": rin.category,
                "operator": op,
                "sliders": sliders,
                "meta": {"error": str(e)},
                **({"dataset_tag": dataset_tag} if dataset_tag else {}),
            })
            continue

        # --- Persist rewriter event
        event = {
            "phase": "rewrite",
            "seed_id": rin.seed_id,
            "category": rin.category,
            "operator": op,
            "sliders": sliders,
            "meta": r.meta,
            **({"dataset_tag": dataset_tag} if dataset_tag else {}),
        }
        if persist_raw:
            event["prompt_text"] = r.text
        rs.append_event(event)

        # --- Optional: ping target model with timing
        if tgt is not None:
            try:
                if verbose:
                    print(f"[M5][{i+1}/{n}] sending to target.generate() ...")
                t2 = time.monotonic()
                out = tgt.generate(r.text, temperature=0.2, max_tokens=256)
                t3 = time.monotonic()
                if verbose:
                    print(f"[M5][{i+1}/{n}] target OK in {(t3 - t2)*1000:.1f} ms | model={out.get('model')} | usage={out.get('usage')}")
                target_text = (out.get("text") or "").strip()
                rs.append_event({
                    "phase": "rewrite_target_echo",
                    "target_id": out.get("model"),
                    "seed_id": rin.seed_id,
                    "operator": op,
                    "target_text": target_text,
                    "response_meta": {
                        "len_tokens": out.get("usage", {}).get("completion_tokens"),
                        "latency_ms": int((t3 - t2) * 1000),
                    },
                    **({"dataset_tag": dataset_tag} if dataset_tag else {}),
                })
            except Exception as e:
                t3 = time.monotonic()
                print(f"[M5][{i+1}/{n}] target ERROR after {(t3 - t2)*1000:.1f} ms: {e}")
                if verbose:
                    traceback.print_exc()
                rs.append_event({
                    "phase": "rewrite_target_error",
                    "seed_id": rin.seed_id,
                    "operator": op,
                    "response_meta": {"error": str(e)},
                    **({"dataset_tag": dataset_tag} if dataset_tag else {}),
                })

    print(
        f"[M5] rewrites={n} raw_saved={persist_raw} target_ping={bool(tgt)} "
        f"(ops={len(ops)}) dataset_tag={dataset_tag or '—'}"
    )

# ----------------------------
# M6: judge
# ----------------------------
@app.command("judge", help="M6: Evaluate rewritten prompts with the Judge LLM and log scores.")
def judge_cmd(
    n: int = typer.Option(10, "--n", "-n", help="Number of rewrites to judge.", show_default=True),
    runs_model: Optional[str] = typer.Option(
        None,
        "--runs-model",
        help="Which model's run folder to read (defaults to first config target).",
    ),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose debug logging."),
):
    _env()
    t0 = _t()
    _p(f"[M6] start | n={n} config_path={config_path}")

    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])
    redact_in_csv = cfg.get("logging", {}).get("redact_in_csv", False)

    model_for_run = runs_model or _resolve_target_model_id(cfg, None) or "unspecified"
    t_rs0 = _t()
    rs = RunStore(runs_root=runs_root, model_name=model_for_run, redact_in_csv=redact_in_csv)
    _p(f"[M6] RunStore in {_ms(_t()-t_rs0)} | run_dir={rs.root}")

    # read events
    if not rs.jsonl_path.exists():
        _p(f"[M6] No events.jsonl at: {rs.jsonl_path}")
        raise typer.Exit(code=1)

    t_read0 = _t()
    rewrites = []
    with rs.jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("phase") == "rewrite" and ev.get("prompt_text"):
                rewrites.append(ev)
    _p(f"[M6] loaded rewrites in {_ms(_t()-t_read0)} | count={len(rewrites)}")

    if not rewrites:
        _p("[M6] No rewrites found. Run M5 first.")
        raise typer.Exit(code=1)

    # build judge
    t_j0 = _t()
    judge = Judge(cfg, rs)
    _p(f"[M6] Judge built in {_ms(_t()-t_j0)} | model={cfg.get('judge',{}).get('model_id')} backend={cfg.get('judge',{}).get('backend')}")

    random.shuffle(rewrites)
    sample = rewrites[: min(n, len(rewrites))]

    judged = 0
    for i, r in enumerate(sample, 1):
        jin = JudgeInput(
            prompt_text=r.get("prompt_text", ""),
            category=r.get("category"),
            seed_id=r.get("seed_id"),
            operator=r.get("operator"),
        )
        _p(f"[M6][{i}/{len(sample)}] judge.start seed_id={jin.seed_id} op={jin.operator}")
        t_call0 = _t()
        try:
            jres = judge.score(jin)
            dt = _t() - t_call0
            judged += 1
            _p(f"[M6][{i}] judge.score OK in {_ms(dt)} | latency_ms={jres.latency_ms} scores={jres.scores}")
            rs.append_event({
                "phase": "judge",
                "seed_id": jin.seed_id,
                "category": jin.category,
                "operator": jin.operator,
                "scores": jres.scores,
                "response_meta": {
                    "latency_ms": jres.latency_ms,
                    "model": jres.model_id,
                },
            })
        except Exception as e:
            dt = _t() - t_call0
            _p(f"[M6][{i}] judge.score ERROR after {_ms(dt)}: {e}")
            if verbose:
                traceback.print_exc()
            rs.append_event({
                "phase": "judge_error",
                "seed_id": jin.seed_id,
                "category": jin.category,
                "operator": jin.operator,
                "response_meta": {"error": str(e)},
            })

    _p(f"[M6] done | judged={judged}/{len(sample)} | total {_ms(_t()-t0)} | run: {rs.root}")

# ----------------------------
# HEALTH: one-shot checks for M1..M6
# ----------------------------
@app.command("health", help="Run milestone health checks (M1..M6) and print PASS/FAIL.")
def health(
    smoke: bool = typer.Option(False, help="Generate 1 rewrite + judge 1 item if missing (auto-fix)."),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
):
    _env()
    from rich.console import Console

    console = Console()

    def ok(tag, msg=""):  console.print(f"✅ {tag} OK {msg}")
    def fail(tag, msg=""): console.print(f"❌ {tag} FAIL {msg}")

    try:
        cfg = load_app_config(config_path)
        paths = cfg["paths"]
        ok("M1", f"[dim]data={paths['data']} artifacts={paths['artifacts']} runs={paths['runs']}[/]")
    except Exception as e:
        fail("M1", f"{e}"); raise typer.Exit(1)

    # M2: data snapshot present?
    try:
        runs_root = Path(cfg["paths"]["runs"])
        # use first target for run name
        tgt = cfg.get("targets", [])[0]
        model_for_run = (tgt.get("id") if isinstance(tgt, dict) else str(tgt)) if tgt else "unspecified"
        redact_in_csv = cfg.get("logging", {}).get("redact_in_csv", False)
        rs = RunStore(runs_root=runs_root, model_name=model_for_run, redact_in_csv=redact_in_csv)
        # If there is no run yet, create one
        if not rs.root.exists() or not (rs.root / "run_meta.json").exists():
            if smoke:
                _ = build_all(Path(cfg["paths"]["data"]), rs.root)
                rs.write_meta(Path(cfg["paths"]["data"]), jbb_category_counts(Path(cfg["paths"]["data"]) / "jbb"))
            else:
                fail("M2", "No run snapshot found; run `python -m src.cli run_init --model <id>` or use --smoke"); raise typer.Exit(1)
        ok("M2", f"[dim]run={rs.root.name}[/]")
    except Exception as e:
        fail("M2", f"{e}"); raise typer.Exit(1)

    # Router/Models
    try:
        rc = RouterClient(base_url=cfg["router"]["base_url"])
        data = rc.list_models()
        models = [m.get("id") or m.get("model") or m for m in (data.get("data") or data.get("models") or [])]
        need = []
        for block in ("targets", "rewriter", "judge"):
            if block == "targets":
                need += [(t["id"] if isinstance(t, dict) else str(t)) for t in cfg.get("targets", [])]
            else:
                mid = cfg.get(block, {}).get("model_id")
                if mid:
                    need.append(mid)
        missing = [m for m in need if m not in models]
        if missing:
            fail("Router", f"Missing on router: {missing}")
        else:
            ok("Router", f"[dim]{len(models)} models visible[/]")
    except Exception as e:
        fail("Router", f"{e}")

    # M3: echo one target
    try:
        router_base = cfg["router"]["base_url"]
        first_target = (cfg["targets"][0]["id"] if isinstance(cfg["targets"][0], dict) else str(cfg["targets"][0])) if cfg.get("targets") else None
        llm = make_llm(model_id=first_target, router_base=router_base)
        tgt = TargetAdapter(llm)
        out = tgt.generate("health check: reply with OK", temperature=0.0, max_tokens=8)
        ok("M3", f"[dim]{out.get('model')} responded[/]")
    except Exception as e:
        fail("M3", f"{e}")

    # M5: at least 1 rewrite exists (or create 1 in smoke)
    try:
        rewrites = []
        with rs.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get("phase") == "rewrite" and ev.get("prompt_text"):
                        rewrites.append(ev)
                except Exception:
                    pass
        if not rewrites and smoke:
            rewriter = PromptRewriter(cfg, rs)
            rin = RewriterInput(
                seed_id="health",
                category="health",
                seed_text="Say OK",
                operator_name="identity",
                sliders={"length": 0.2, "temperature": 0.2, "persona": 0.0},
            )
            r = rewriter.rewrite(rin)
            rs.append_event({
                "phase": "rewrite",
                "seed_id": "health",
                "category": "health",
                "operator": "identity",
                "prompt_text": r.text,
                "meta": r.meta,
            })
            rewrites = [r]
        if rewrites:
            ok("M5", f"[dim]{len(rewrites)} rewrite(s) present[/]")
        else:
            fail("M5", "No rewrites; run `rewrite-sample` or use --smoke")
    except Exception as e:
        fail("M5", f"{e}")

    # M6: score exactly 1 item
    try:
        judge = Judge(cfg, rs)
        # pick first real rewrite
        sample = None
        with rs.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get("phase") == "rewrite" and ev.get("prompt_text"):
                        sample = ev
                        break
                except Exception:
                    pass
        if not sample:
            fail("M6", "No rewrite to judge; run with --smoke"); return
        jin = JudgeInput(
            prompt_text=sample.get("prompt_text", "OK"),
            category=sample.get("category"),
            seed_id=sample.get("seed_id"),
            operator=sample.get("operator"),
        )
        jres = judge.score(jin)
        rs.append_event({
            "phase": "judge",
            "seed_id": jin.seed_id,
            "category": jin.category,
            "operator": jin.operator,
            "scores": jres.scores,
            "response_meta": {
                "latency_ms": jres.latency_ms, 
                "model": jres.model_id,
        },
            #"dataset_tag": sample.get("dataset_tag"),  # optional for health
            #"pair_id": f"{jin.seed_id}::{jin.operator}",
})          
        ok("M6", f"[dim]{jres.model_id} scored {jres.scores}[/]")
    except Exception as e:
        fail("M6", f"{e}")

# ----------------------------
# M7: reward
# ----------------------------
@app.command("reward", help="M7: Convert judge scores into scalar rewards and log them (with judge backfill).")
def reward_cmd(
    runs_model: Optional[str] = typer.Option(None, help="Which run to read (defaults to first config target)."),
    config_path: str = "config/stack.yaml",
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose debug logging."),
    backfill: bool = typer.Option(True, "--backfill/--no-backfill", help="Backfill missing judge events before scoring."),
):
    _env()
    t0 = _t()
    _p(f"[M7] start | backfill={backfill} config_path={config_path}")

    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])

    def _resolve_target_model_id_local(cfg: dict) -> Optional[str]:
        targets = cfg.get("targets", [])
        if not targets: return None
        first = targets[0]
        return first.get("id") if isinstance(first, dict) else str(first)

    model_for_run = runs_model or _resolve_target_model_id_local(cfg) or "unspecified"
    t_rs0 = _t()
    rs = RunStore(runs_root=runs_root, model_name=model_for_run)
    _p(f"[M7] RunStore in {_ms(_t()-t_rs0)} | run_dir={rs.root}")

    if not rs.jsonl_path.exists():
        _p(f"[M7] No events.jsonl at: {rs.jsonl_path}")
        raise typer.Exit(code=1)

    # Pass 1: scan events
    t_scan0 = _t()
    rewrites = []
    judges_by_pair = {}
    calls_by_pair = {}
    target_text_by_pair = {}

    import hashlib
    def hsh(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    with rs.jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            ph = ev.get("phase")
            if ph == "rewrite" and ev.get("prompt_text"):
                txt = ev["prompt_text"]
                key = (ev.get("seed_id"), ev.get("operator"), hsh(txt))
                rewrites.append((key, ev))
            elif ph == "judge":
                pair = (ev.get("seed_id"), ev.get("operator"))
                judges_by_pair[pair] = ev
            elif ph == "rewrite_target_echo":
                pair = (ev.get("seed_id"), ev.get("operator"))
                calls_by_pair[pair] = int(calls_by_pair.get(pair, 0)) + 1
                tt = ev.get("target_text") or ""
                if isinstance(tt, str) and tt.strip():
                    target_text_by_pair[pair] = tt

    _p(f"[M7] scan events in {_ms(_t()-t_scan0)} | rewrites={len(rewrites)} judged_pairs={len(judges_by_pair)} target_calls_pairs={len(calls_by_pair)}")

    # Backfill missing judge
    if backfill:
        t_bf0 = _t()
        latest_rewrite_for_pair = {}
        for (seed_id, op, _), r_ev in rewrites:
            latest_rewrite_for_pair[(seed_id, op)] = r_ev

        judge = Judge(cfg, rs)
        new_judges = 0
        for pair, r_ev in latest_rewrite_for_pair.items():
            if pair in judges_by_pair:
                continue
            seed_id, operator = pair
            prompt = r_ev.get("prompt_text", "")
            category = r_ev.get("category")
            target_reply = target_text_by_pair.get(pair)
            try:
                jres = judge.score(JudgeInput(
                    prompt_text=prompt,
                    category=category,
                    seed_id=seed_id,
                    operator=operator,
                    target_response=target_reply,
                ))
                rs.append_event({
                    "phase": "judge",
                    "seed_id": seed_id,
                    "category": category,
                    "operator": operator,
                    "scores": jres.scores,
                    "response_meta": {
                        "latency_ms": jres.latency_ms,
                        "model": jres.model_id,
                        "parsed_ok": getattr(jres, "parsed_ok", None),
                        "backfill": True,
                    },
                })
                judges_by_pair[pair] = {"scores": jres.scores}
                new_judges += 1
            except Exception as e:
                rs.append_event({
                    "phase": "judge_error",
                    "seed_id": seed_id,
                    "category": category,
                    "operator": operator,
                    "response_meta": {"error": str(e), "during": "backfill"},
                })
                if verbose:
                    traceback.print_exc()
        _p(f"[M7] backfill in {_ms(_t()-t_bf0)} | new_judges={new_judges}")

    # Rewards
    t_rw0 = _t()
    calc = RewardCalculator(cfg)
    wrote = 0
    seen_text_hashes = set()

    for (seed_id, operator, p_hash), r_ev in rewrites:
        prompt = r_ev.get("prompt_text", "")
        pair = (seed_id, operator)
        j_ev = judges_by_pair.get(pair)
        if not j_ev:
            if verbose:
                _p(f"[M7] skip: no judge for seed={seed_id} op={operator}")
            continue

        scores = j_ev.get("scores", {}) or {}
        calls = int(calls_by_pair.get(pair, 0))
        length = len(prompt)
        seen_before = p_hash in seen_text_hashes
        seen_text_hashes.add(p_hash)

        rin = RewardInput(
            prompt_text=prompt,
            scores=scores,
            calls=calls,
            seen_before=seen_before,
            length=length,
        )
        r = calc.compute(rin)

        rs.append_event({
            "phase": "reward",
            "seed_id": seed_id,
            "category": r_ev.get("category"),
            "operator": operator,
            "reward": r.reward,
            "base_score": r.base_score,
            "penalties": r.penalties,
            "detail": r.detail,
        })
        wrote += 1

        if verbose:
            _p(f"[M7] reward: seed={seed_id} op={operator} base={r.base_score:.3f} -> R={r.reward:.3f} calls={calls} seen_before={seen_before}")

    _p(f"[M7] rewards pass in {_ms(_t()-t_rw0)} | rewards_written={wrote}")
    _p(f"[M7] done | total {_ms(_t()-t0)} | run: {rs.root}")

# ----------------------------
# M8: Train SAC (env passthrough for OOD)
# ----------------------------
@app.command("train-sac", help="M8: Train Hybrid SAC on PromptHybrid-v0 (checkpoints + CSV metrics).")
def train_sac(
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    runs_model: Optional[str] = typer.Option(None, "--runs-model", help="Run folder key (defaults to first target)."),
    total_steps: int = typer.Option(2000, "--total-steps", "-t"),
    seed: int = typer.Option(1, "--seed"),
    learning_starts: int = typer.Option(256, "--learning-starts"),
    batch_size: int = typer.Option(64, "--batch-size"),
    alpha: float = typer.Option(0.2, "--alpha"),
    save_every: int = typer.Option(500, "--save-every"),
    log_every: int = typer.Option(50, "--log-every"),
    resume_from: Optional[str] = typer.Option(None, "--resume-from"),
    # model overrides for the env
    rewriter_model: Optional[str] = typer.Option(None, "--rewriter-model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    target_model: Optional[str] = typer.Option(None, "--target-model"),
    use_target_echo: bool = typer.Option(False, "--use-target-echo", help="Ping target once per step."),
    # NEW: OOD passthrough
    seeds_path_override: Optional[str] = typer.Option(None, "--seeds-path-override", help="JSONL file for OOD seeds."),
    dataset_tag: Optional[str] = typer.Option(None, "--dataset-tag", help='Tag env events, e.g. "OOD-JBB".'),
):
    _env()
    t0 = _t()
    _p(f"[CLI][train-sac] start | config_path={config_path}")

    from src.rl.hybrid_sac import train_hybrid_sac, SACConfig
    from src.store.run_store import RunStore
    from src.config import load_app_config

    t_cfg0 = _t()
    cfg_all = load_app_config(config_path)
    _p(f"[CLI][train-sac] loaded app config in {_ms(_t()-t_cfg0)}")

    runs_root = Path(cfg_all["paths"]["runs"])

    def _resolve_target(cfg: dict) -> Optional[str]:
        t = cfg.get("targets", [])
        if not t: return None
        first = t[0]
        return first.get("id") if isinstance(first, dict) else str(first)

    model_for_run = runs_model or _resolve_target(cfg_all) or "unspecified"
    _p(f"[CLI][train-sac] model_for_run={model_for_run}")

    t_rs0 = _t()
    rs = RunStore(runs_root=runs_root, model_name=model_for_run)
    _p(f"[CLI][train-sac] RunStore ready in {_ms(_t()-t_rs0)} | run_dir={rs.root}")

    out_dir = (Path(rs.root) / "rl_sac").as_posix()
    _p(f"[CLI][train-sac] out_dir={out_dir}")

    sac_cfg = SACConfig(
        total_steps=total_steps,
        learning_starts=learning_starts,
        batch_size=batch_size,
        alpha=alpha,
        save_every=save_every,
        log_every=log_every,
        seed=seed,
    )
    _p(f"[CLI][train-sac] SACConfig: {sac_cfg}")

    env_kwargs = {
        "config_path": config_path,
        "runs_model": runs_model,
        "rewriter_model": rewriter_model,
        "judge_model": judge_model,
        "target_model": target_model,
        "use_target_echo": use_target_echo,
        "seeds_path_override": seeds_path_override,
        "dataset_tag": dataset_tag,
    }
    _p(f"[CLI][train-sac] env_kwargs={env_kwargs}")

    t_tr0 = _t()
    _p("[CLI][train-sac] calling train_hybrid_sac(...)")
    res = train_hybrid_sac(
        env_id="PromptHybrid-v0",
        env_kwargs=env_kwargs,
        cfg=sac_cfg,
        out_dir=out_dir,
        resume_from=resume_from,
    )
    _p(f"[CLI][train-sac] train_hybrid_sac returned in {_ms(_t()-t_tr0)}")

    _p(f"[SAC] training complete. Out: {res['out_dir']}")
    _p(f"[CLI][train-sac] total runtime {_ms(_t()-t0)}")

# ----------------------------
# M8: Evaluate SAC (env passthrough for OOD)
# ----------------------------
@app.command("eval-sac", help="M8: Evaluate a trained SAC checkpoint (greedy policy).")
def eval_sac(
    checkpoint: str = typer.Option(..., "--checkpoint"),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    runs_model: Optional[str] = typer.Option(None, "--runs-model"),
    rewriter_model: Optional[str] = typer.Option(None, "--rewriter-model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    target_model: Optional[str] = typer.Option(None, "--target-model"),
    use_target_echo: bool = typer.Option(False, "--use-target-echo"),
    episodes: int = typer.Option(20, "--episodes"),
    seeds_path_override: Optional[str] = typer.Option(None, "--seeds-path-override"),
    dataset_tag: Optional[str] = typer.Option(None, "--dataset-tag"),
    progress_every: int = typer.Option(0, "--progress-every", help="Print progress every N episodes (0=off)."),
):
    _env()
    from src.rl.hybrid_sac import eval_hybrid_sac
    env_kwargs = {
        "config_path": config_path,
        "runs_model": runs_model,
        "rewriter_model": rewriter_model,
        "judge_model": judge_model,
        "target_model": target_model,
        "use_target_echo": use_target_echo,
        "seeds_path_override": seeds_path_override,
        "dataset_tag": dataset_tag,
    }
    out = eval_hybrid_sac(
        checkpoint_path=checkpoint,
        env_id="PromptHybrid-v0",
        env_kwargs=env_kwargs,
        episodes=episodes,
        progress_every=progress_every,
    )
    print(f"[EVAL] episodes={out['episodes']} mean_reward={out['mean_reward']:.4f} ± {out['std_reward']:.4f}")

# ----------------------------
# M8: Rollout top-k (env passthrough for OOD)
# ----------------------------
@app.command("rollout-topk", help="M8: Rollout greedy policy and export prompts (one-step env).")
def rollout_topk(
    checkpoint: str = typer.Option(..., "--checkpoint"),
    out_csv: str = typer.Option("artifacts/rl/rollout_topk.csv", "--out-csv"),
    n: int = typer.Option(50, "--n"),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    runs_model: Optional[str] = typer.Option(None, "--runs-model"),
    rewriter_model: Optional[str] = typer.Option(None, "--rewriter-model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    target_model: Optional[str] = typer.Option(None, "--target-model"),
    use_target_echo: bool = typer.Option(False, "--use-target-echo"),
    # NEW
    seeds_path_override: Optional[str] = typer.Option(None, "--seeds-path-override"),
    dataset_tag: Optional[str] = typer.Option(None, "--dataset-tag"),
):
    _env()
    from src.rl.hybrid_sac import rollout_and_export
    env_kwargs = {
        "config_path": config_path,
        "runs_model": runs_model,
        "rewriter_model": rewriter_model,
        "judge_model": judge_model,
        "target_model": target_model,
        "use_target_echo": use_target_echo,
        "seeds_path_override": seeds_path_override,
        "dataset_tag": dataset_tag,
    }
    path = rollout_and_export(checkpoint_path=checkpoint, env_id="PromptHybrid-v0", env_kwargs=env_kwargs, n=n, out_csv=out_csv)
    print(f"[ROLLOUT] wrote: {path}")
# ----------------------------
# M10: Minimal OOD metrics
# ----------------------------
@app.command("metrics-ood", help="M10: Compute OOD metrics from events (filter by --dataset-tag).")
def metrics_ood(
    dataset_tag: str = typer.Option(..., "--dataset-tag", help='Tag used during generation, e.g. "OOD-JBB".'),
    runs_model: Optional[str] = typer.Option(None, "--runs-model", help="Run folder key (defaults to first target)."),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
    success_threshold: float = typer.Option(0.65, "--success-threshold", help="Judge success threshold."),
    out_path: Optional[str] = typer.Option(None, "--out", help="Write JSON metrics to this path."),
):
    _env()
    from rich.table import Table
    from src.eval.metrics import compute_metrics

    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])
    model_for_run = runs_model or _resolve_target_model_id(cfg, None) or "unspecified"
    rs = RunStore(runs_root=runs_root, model_name=model_for_run)

    metrics = compute_metrics(
        rs.jsonl_path,
        dataset_tag=dataset_tag,
        success_threshold=success_threshold,
    )
    if "error" in metrics:
        print(f"[M10] {metrics['error']}")
        raise typer.Exit(code=1)

    # pretty print
    m = metrics["metrics"]
    counts = metrics["counts"]
    table = Table("Metric", "Value")
    table.add_row("pairs", str(counts["pairs"]))
    table.add_row("successes", str(counts["successes"]))
    table.add_row("total_target_calls", str(counts["total_target_calls"]))
    table.add_row("ASR", f"{m['ASR']:.3f}")
    table.add_row("QueriesPerSuccess", f"{m['QueriesPerSuccess']:.3f}" if m["QueriesPerSuccess"] != float('inf') else "inf")
    table.add_row("Distinct-1", f"{m['Distinct1']:.3f}")
    table.add_row("Distinct-2", f"{m['Distinct2']:.3f}")
    table.add_row("Distinct-3", f"{m['Distinct3']:.3f}")
    print(table)

    # optional aspect means
    jm = metrics.get("judge_means", {})
    if jm:
        t2 = Table("Judge aspect", "Mean")
        for k in sorted(jm.keys()):
            try:
                t2.add_row(str(k), f"{float(jm[k]):.3f}")
            except Exception:
                t2.add_row(str(k), str(jm[k]))
        print(t2)

    # write JSON
    if out_path:
        outp = Path(out_path)
    else:
        outp = Path(cfg["paths"]["artifacts"]) / "eval" / rs.root.name / f"metrics_{dataset_tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M10] wrote: {outp}")

# ----------------------------
# Train all targets on ID (SORRY)
# ----------------------------
@app.command("train-all", help="Train SAC once per configured target on ID/SORRY seeds.")
def train_all(
    total_steps: int = typer.Option(20000, "--total-steps"),
    use_target_echo: bool = typer.Option(True, "--use-target-echo"),
    rewriter_model: Optional[str] = typer.Option(None, "--rewriter-model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
):
    _env()
    cfg = load_app_config(config_path)
    targets = [t["id"] if isinstance(t, dict) else str(t) for t in cfg.get("targets", [])]
    runs_root = Path(cfg["paths"]["runs"])

    for t in targets:
        rs = RunStore(runs_root=runs_root, model_name=t)
        # ensure M2 snapshot exists for this target
        if not (rs.root / "seeds.jsonl").exists():
            typer.echo(f"[train-all] No snapshot for {t}. Running run_init…")
            run_init.callback(model=t, config_path=config_path)

        typer.echo(f"[train-all] target={t} (ID-SORRY)")
        train_sac.callback(
            config_path=config_path,
            runs_model=t,
            total_steps=total_steps,
            use_target_echo=use_target_echo,
            rewriter_model=rewriter_model,
            judge_model=judge_model,
            target_model=t,
            seeds_path_override=None,        # ID training → use SORRY seeds from run snapshot
            dataset_tag="ID-SORRY",
        )

# ----------------------------
# Test all targets on OOD (JBB)
# ----------------------------
@app.command("test-all", help="Evaluate latest SAC checkpoint for all targets on OOD/JBB and write metrics.")
def test_all(
    episodes: int = typer.Option(200, "--episodes"),
    config_path: str = typer.Option("config/stack.yaml", "--config-path"),
):
    _env()
    cfg = load_app_config(config_path)
    runs_root = Path(cfg["paths"]["runs"])
    targets = [t["id"] if isinstance(t, dict) else str(t) for t in cfg.get("targets", [])]

    for t in targets:
        rs = RunStore(runs_root=runs_root, model_name=t)

        # make sure OOD seeds exist for this run
        ood_path = rs.root / "seeds_ood.jbb.jsonl"
        if not ood_path.exists():
            typer.echo(f"[test-all] exporting OOD for {t}")
            export_ood.callback(n=10**9, runs_model=t, config_path=config_path)

        # pick latest checkpoint
        ck_dir = rs.root / "rl_sac" / "checkpoints"
        if not ck_dir.exists():
            typer.echo(f"[test-all] no checkpoints for {t} — did you train?"); continue
        ck = max(ck_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)

        typer.echo(f"[test-all] target={t} ckpt={ck.name} (OOD-JBB)")
        eval_sac.callback(
            checkpoint=ck.as_posix(),
            config_path=config_path,
            runs_model=t,
            target_model=t,
            use_target_echo=True,
            seeds_path_override=ood_path.as_posix(),
            dataset_tag="OOD-JBB",
            episodes=episodes,
        )
        metrics_ood.callback(dataset_tag="OOD-JBB", runs_model=t, config_path=config_path)


# ----------------------------
if __name__ == "__main__":
    app()
