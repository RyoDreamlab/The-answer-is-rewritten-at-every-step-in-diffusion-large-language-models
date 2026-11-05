#!/usr/bin/env python3
"""MMStar/POPE stock/gated/gray/prior decodes for LLaDA-V only (new loader 2026-10-22).
New dated file 2026-10-22; imports (never edits) calib_decode_multi_20261013e/c.
Datasets: vstar datasets/vstar_20261022/manifest.json (191, 4 options), mme datasets/mme_20261022/manifest.json (2374, Yes/No as 2 labels).
MME mapping: Yes/No treated exactly as POPE yesno task in 20261013e (labels Yes/No single tokens, gated shift on 2 label logits,
prior = pope_text('N/A')). This is a Yes/No-as-two-options adaptation, NOT the paper's original letter protocol, and NOT
Thinking Diffusion's lmms-eval mme scoring (their MME scoring aggregates per subtask, existence/count/position/color + total).
Outputs only under outputs/*_20261022/; resumable by id (e.run_decode), flock per output.
"""
from __future__ import annotations
import argparse, fcntl, json, math, os, sys, time
from pathlib import Path
import torch
PROJECT=Path(os.environ.get("URD_PROJECT","/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0,str(PROJECT/"experiments"/"unmask_order")); sys.path.insert(0,str(PROJECT/"experiments"))
import calib_decode_multi_20261013c as c
import calib_decode_multi_20261013e as e
import run_unmask_order as base
GEN=base.GEN; LETTERS=e.LETTERS; YESNO=e.YESNO

def _load_manifest(name):
    return json.loads((PROJECT/f"datasets/{name}_20261022/manifest.json").read_text(encoding="utf-8"))

def rows_mmstar(backend,limit,image_cond):
    return e.rows_mmstar(backend,limit,image_cond)
def rows_pope(backend,limit,image_cond):
    return e.rows_pope(backend,limit,image_cond)
DATASETS={"mmstar":rows_mmstar,"pope":rows_pope}
TASK={"mmstar":"mcq","pope":"yesno"}

class LladavE(c.Lladav):
    def decode(self,text,image):
        pe,plen=self.lb.prefix(self.ctx,text,image)
        x=torch.full((1,GEN),self.mask_id,dtype=torch.long,device=pe.device)
        d=e.decode_canvas(x,self.mask_id,self.tokenizer,self.lb.make_forward(self.ctx,pe))
        d["prefix_len"]=plen; return d

BACKENDS={"lladav":LladavE}
DEFAULT_MODEL={"lladav":"checkpoints/LLaDA-V"}

def prior_text(backend,dataset,n):
    if dataset=="pope": return e.pope_text("N/A")
    return c.text_for(backend,"N/A","",["N/A"]*n)

def run_prior(args,be):
    # clone of e.run_prior with correct ns for yesno (2,) vs mcq (2..5)
    head,gamma64,eps=be.prepare_readout(); ids=e.STATE["label_ids"]; labels=e.STATE["labels"]
    W64=torch.stack([head[i].detach().to(torch.float64).cpu() for i in ids])
    ns=(2,) if TASK[args.dataset]=="yesno" else (2,3,4,5)
    records=[]
    for n in ns:
        text=prior_text(be.name,args.dataset,n)
        with torch.no_grad(): logits,h,plen,prompt=be.prior_forward(text,c.gray_image())
        h64=h.to(torch.float64); l64=[float(v) for v in (W64@(gamma64*(h64*torch.rsqrt(torch.dot(h64,h64)/h64.shape[0]+eps))))]
        lm=[float(logits[i]) for i in ids]; lp=torch.log_softmax(torch.tensor(l64[:n],dtype=torch.float64),dim=0)
        a=int(torch.argmax(logits).item())
        rec={"id":f"global_n{n}","cond":"a_global","backend":be.name,"dataset":args.dataset,"n_opt":n,"labels":labels[:n],"prompt_text":prompt,"letter_ids":ids,"letter_logit64":l64,"letter_logit_model":lm,"max_abs_diff_readout_vs_model":max(abs(x-y) for x,y in zip(l64,lm)),"argmax_id":a,"argmax_text":base.piece(be.tokenizer,a,be.mask_id),"prefix_len":plen,"decode_step":0,"canvas_index":0,"p_cf":[float(v) for v in lp.exp()],"log_p_cf":[float(v) for v in lp],"alpha_1_minus_pmin_over_pmax":1.0-math.exp(-float(lp.max()-lp.min()))}
        records.append(rec); print(f"PRIOR backend={be.name} dataset={args.dataset} n={n} p_cf={dict(zip(labels,[round(v,4) for v in rec['p_cf']]))} alpha={rec['alpha_1_minus_pmin_over_pmax']:.6f}",flush=True)
    by_n={str(r["n_opt"]):{"p_cf":r["p_cf"],"log_p_cf":r["log_p_cf"],"alpha_spread":r["alpha_1_minus_pmin_over_pmax"],"prior_label":labels[int(torch.tensor(r["p_cf"]).argmax())]} for r in records}
    out={"backend":be.name,"dataset_template_of":args.dataset,"labels":labels,"letter_ids":ids,"gray":c.GRAY,"image_size":c.GLOBAL_SIZE,"gen":GEN,"rule":"alpha_n = 1 - p_min/p_max over the n labels of cond a_global","by_n_opt":by_n,"records":records}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding="utf-8"); print(f"wrote {args.output}",flush=True)

def check_output_path(q):
    r=q.resolve()
    if (PROJECT/"outputs").resolve() not in r.parents or not r.parent.name.endswith("_20261022"):
        raise RuntimeError(f"refusing to write outside outputs/*_20261022/: {q}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=tuple(BACKENDS),required=True); p.add_argument("--dataset",choices=tuple(DATASETS),required=True)
    p.add_argument("--mode",choices=("stock","gated","gray","prior"),required=True); p.add_argument("--alpha-rule",choices=("spread","full"),default="spread")
    p.add_argument("--alpha-scale",type=float,default=1.0); p.add_argument("--prior-file",type=Path,default=None); p.add_argument("--cuda",type=int,required=True)
    p.add_argument("--limit",type=int,default=0); p.add_argument("--output",type=Path,required=True); p.add_argument("--progress",type=Path,default=None)
    p.add_argument("--model-path",default=""); p.add_argument("--vq-model",default=str(PROJECT/"checkpoints/magvitv2")); p.add_argument("--mmada-repo",default=str(PROJECT/"repos/MMaDA")); p.add_argument("--print-prompts",type=int,default=0)
    args=p.parse_args(); image_cond="gray" if args.mode=="gray" else "real"
    if args.print_prompts:
        for row in DATASETS[args.dataset](args.backend,args.print_prompts,image_cond):
            print(json.dumps({"id":row["id"],"gold":row["gold"],"n_opt":row["n_opt"],"text":row["text"]},ensure_ascii=False,indent=1))
        return
    if args.cuda not in (0,1): raise RuntimeError("--cuda must be 0 or 1")
    for q in (args.output,args.progress):
        if q is not None: check_output_path(q)
    args.model_path=args.model_path or str(PROJECT/DEFAULT_MODEL[args.backend])
    e.STATE["mode"]=args.mode; e.STATE["alpha_scale"]=float(args.alpha_scale)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=="prior" and args.output.exists(): raise RuntimeError(f"prior exists {args.output}")
    lock=args.output.with_suffix(args.output.suffix+".lock").open("a",encoding="utf-8")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError(f"already running {args.output}") from exc
    task=TASK[args.dataset]; labels=list(LETTERS) if task=="mcq" else list(YESNO)
    if args.mode=="gated":
        if args.prior_file is None: raise RuntimeError("--prior-file required")
        log_pcf,alpha,pcf,plabels=e.load_priors(args.prior_file)
        if plabels!=labels: raise RuntimeError(f"prior labels {plabels} != {labels}")
        if args.alpha_rule=="full": alpha={n:1.0 for n in alpha}
        e.STATE["log_pcf"],e.STATE["alpha"]=log_pcf,alpha
        print(f"PRIOR loaded {args.prior_file} alpha={alpha}",flush=True)
    torch.cuda.set_device(args.cuda); torch.set_grad_enabled(False)
    print(f"start backend={args.backend} dataset={args.dataset} mode={args.mode} cuda={args.cuda} limit={args.limit} output={args.output}",flush=True)
    rows=None
    if args.mode!="prior":
        rows=DATASETS[args.dataset](args.backend,args.limit,image_cond)
        need={r["n_opt"] for r in rows}
        if args.mode=="gated" and not need<=set(e.STATE["log_pcf"]): raise RuntimeError(f"prior missing {need}")
        print(f"rows={len(rows)} n_opt_hist={{ {need} }} example:\n{rows[0]['text']}",flush=True)
    be=BACKENDS[args.backend](args); e.set_labels(be,task)
    if args.mode=="prior": run_prior(args,be); return
    if args.progress is None: raise RuntimeError("--progress required")
    import letter_argmax_traj as traj; traj.repair_tail(args.output)
    e.run_decode(args,be,rows)

if __name__=="__main__": main()
