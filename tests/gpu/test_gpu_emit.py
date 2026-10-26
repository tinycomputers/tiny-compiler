import copy
import json
import re

import jsonschema
import pytest

from tests.gpu.helpers import EXAMPLES, ROOT, build
from tinyc.gpu.emit import dumps

SCHEMA = json.loads((ROOT / "docs" / "launch.schema.json").read_text())


def validate(doc):
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)


def test_the_schema_itself_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1, 2])
def test_every_launch_document_matches_the_schema(path, opt):
    comp = build(path.read_text(), opt=opt)
    validate(comp.launch)
    assert comp.launch["estimate"] == comp.program.estimate


@pytest.mark.parametrize("opt", [0, 1, 2])
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_tgpu_assembly_golden(path, opt, golden):
    golden(
        ROOT / "tests" / "golden" / "tgpu" / f"O{opt}" / f"{path.stem}.tgpu",
        build(path.read_text(), opt=opt).program.asm_text(),
    )


def test_tiled_kernels_declare_their_shared_memory_in_the_launch_document():
    comp = build("input A : i8[4, 8]\ninput B : i8[8, 4]\nC = A @ B\noutput C\n", opt=2)
    (k,) = comp.launch["kernels"]
    assert k["shared_size"] == 8 and "SYNCB" in k["asm"] and "SHLD" in k["asm"]
    assert "8 shared words per block" in k["asm"]
    assert comp.launch["estimate"]["per_kernel"][0]["shared_size"] == 8


def test_emission_is_deterministic():
    src = (ROOT / "examples" / "mlp_gpu.tiny").read_text()
    assert dumps(build(src).launch) == dumps(build(src).launch)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_instruction_carries_its_source_position(path):
    for asm in build(path.read_text()).program.asm.values():
        for line in asm.splitlines():
            if line and not line.startswith(";") and not line.endswith(":"):
                assert re.search(r"; \d+:\d+$", line), line


def test_header_comments_and_labels_are_valid_for_the_assembler():
    asm = build("input A : i8[2, 2]\ninput B : i8[2, 2]\nC = relu(A @ B)\noutput C\n").program.asm[
        "k0_matmul"
    ]
    assert asm.startswith("; kernel k0_matmul") and "reserved" in asm.splitlines()[1]
    labels = [ln for ln in asm.splitlines() if ln.endswith(":")]
    assert labels == ["L0:", "L1:"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("version", 2),
        lambda d: d.__setitem__("num_registers", 4),
        lambda d: d.__setitem__("extra", 1),
        lambda d: d["kernels"][0].__setitem__("op", "conv"),
        lambda d: d["kernels"][0].__setitem__("grid", [4]),
        lambda d: d["kernels"][0].pop("asm"),
        lambda d: d["buffers"]["A"].__setitem__("dtype", "f32"),
        lambda d: d["buffers"]["A"].__setitem__("addr", -1),
        lambda d: d.__setitem__("outputs", []),
        lambda d: d["estimate"]["cycles"].pop("max"),
    ],
)
def test_the_schema_rejects_malformed_documents(mutate):
    doc = copy.deepcopy(build((ROOT / "examples" / "mm_relu.tiny").read_text()).launch)
    validate(doc)
    mutate(doc)
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_a_program_with_no_ops_is_a_valid_launch_with_zero_kernels():
    comp = build("const W : i8[2, 2] = [[1, 2], [3, 4]]\noutput W\n")
    validate(comp.launch)
    assert comp.launch["kernels"] == [] and comp.launch["buffers"]["0"]["kind"] == "const"
