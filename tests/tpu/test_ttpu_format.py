import copy
import json

import jsonschema
import numpy as np
import pytest

from tests.tpu.helpers import ROOT, TPU_EXAMPLES, ZERO_K_TILE, build, check_against_oracle
from tinyc.ir import ref
from tinyc.tpu import interp
from tinyc.tpu.cost import estimate
from tinyc.tpu.emit import dumps

SCHEMA = json.loads((ROOT / "docs" / "ttpu.schema.json").read_text())


def validate(doc):
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)


def test_the_schema_itself_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("path", TPU_EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1])
def test_every_emitted_program_matches_the_schema(path, opt):
    comp = build(path.read_text(), opt=opt)
    validate(comp.program)
    assert comp.program["estimate"] == estimate(comp.schedule)


def test_emitted_json_is_deterministic_and_round_trips():
    a = build(ZERO_K_TILE).program
    assert dumps(a) == dumps(build(ZERO_K_TILE).program)
    assert json.loads(dumps(a)) == a


@pytest.mark.parametrize("path", TPU_EXAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("opt", [0, 1])
def test_reference_interpreter_agrees_with_the_oracle(path, opt):
    check_against_oracle(build(path.read_text(), opt=opt))


def test_interpreter_counts_what_the_estimate_predicts():
    for path in TPU_EXAMPLES:
        comp = build(path.read_text())
        inputs = ref.random_inputs(comp.func, np.random.default_rng(3))
        run = interp.run_program(comp.program, inputs)
        assert run["passes"] == comp.program["estimate"]["passes"]
        assert run["host_ops"] == comp.program["estimate"]["host_ops"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("version", 2),
        lambda d: d.__setitem__("tile", 8),
        lambda d: d.__setitem__("extra", 1),
        lambda d: d["commands"].append({"op": "frobnicate"}),
        lambda d: d["commands"][1].pop("id"),
        lambda d: d["commands"][1]["a"].__setitem__("r", -1),
        lambda d: d["commands"][1].__setitem__("surprise", True),
        lambda d: d["buffers"]["A"].__setitem__("dtype", "f32"),
        lambda d: d["buffers"]["A"].__setitem__("shape", [1, 2, 3]),
        lambda d: d.__setitem__("outputs", []),
        lambda d: d["estimate"].pop("rtl_cycles"),
    ],
)
def test_the_schema_rejects_malformed_programs(mutate):
    doc = copy.deepcopy(build((ROOT / "examples" / "mm_relu.tiny").read_text()).program)
    validate(doc)
    mutate(doc)
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
