"""Graph-level passes and the pass manager."""

from tinyc.passes.canonicalize import Canonicalize
from tinyc.passes.const_fold import ConstFold
from tinyc.passes.cse import CSE
from tinyc.passes.dce import DCE
from tinyc.passes.fuse import Fuse
from tinyc.passes.legalize_gpu import LegalizeGpu
from tinyc.passes.legalize_tpu import LegalizeTpu
from tinyc.passes.manager import Fixpoint, PassContext, PassManager, PassReport, PassVerifyError
from tinyc.passes.range_check import RangeCheck
from tinyc.passes.shape_infer import ShapeInfer

REGISTRY = {
    p.name: p
    for p in (
        ShapeInfer, Canonicalize, ConstFold, CSE, DCE, Fuse, RangeCheck, LegalizeTpu, LegalizeGpu
    )
}  # fmt: skip

__all__ = ["REGISTRY", "Fixpoint", "PassContext", "PassManager", "PassReport", "PassVerifyError"]
