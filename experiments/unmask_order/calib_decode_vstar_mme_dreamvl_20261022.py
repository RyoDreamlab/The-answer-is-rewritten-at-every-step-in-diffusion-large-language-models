#!/usr/bin/env python3
"""Dream-VL wrapper for V*/MME (2026-10-22, new file). Monkey-patches calib_decode_dreamvl_20261013g DATASETS/TASK/check_output_path and e.run_prior (yesno ns fix), then calls its main(). Never edits originals."""
import os, sys
from pathlib import Path
PROJECT=Path(os.environ.get("URD_PROJECT","/mnt/pfs/peng_gao/visual-hysteresis"))
sys.path.insert(0,str(PROJECT/"experiments"/"unmask_order")); sys.path.insert(0,str(PROJECT/"experiments"))
import calib_decode_multi_20261013c as c
import calib_decode_multi_20261013e as e
import calib_decode_dreamvl_20261013g as g
import calib_decode_vstar_mme_20261022 as vm

def _wrap(fn):
    def f(backend,limit,image_cond): return g._with_gray(fn(backend,limit,image_cond),image_cond)
    return f
g.DATASETS={"vstar":_wrap(vm.rows_vstar),"mme":_wrap(vm.rows_mme)}
g.TASK={"vstar":"mcq","mme":"yesno"}
def _check(q):
    r=q.resolve()
    if (PROJECT/"outputs").resolve() not in r.parents or "_20261022" not in str(r):
        raise RuntimeError(f"refusing outside outputs/*20261022: {q}")
g.check_output_path=_check
e.run_prior=vm.run_prior
# vm.run_prior uses TASK from vm (same mapping) - ok
if __name__=="__main__": g.main()
