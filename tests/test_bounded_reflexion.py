import pytest

from src.agent import BoundedReflexion


def climbing(scores):
    it = iter(scores)

    def judge(task, output):
        score = next(it, scores[-1])
        return score, ["Do better."] if score < 0.9 else []

    return judge


def gen(task, constraints):
    return f"answer with {len(constraints)} constraints"


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        BoundedReflexion(generate=gen, judge=climbing([1.0]), min_delta=5)
    with pytest.raises(ValueError):
        BoundedReflexion(generate=gen, judge=climbing([1.0]), max_revisions=-1)


def test_single_attempt_when_no_headroom():
    r = BoundedReflexion(generate=gen, judge=climbing([0.5, 0.5]), max_revisions=0)
    out = r("task")
    assert len(out.attempts) == 1 and out.stopped_because == "revision_cap"


def test_respects_revision_cap():
    r = BoundedReflexion(generate=gen, judge=climbing([0.1, 0.3, 0.5, 0.7]), max_revisions=3)
    assert len(r("task").attempts) == 4


def test_plateau_stops_early():
    r = BoundedReflexion(generate=gen, judge=climbing([0.70, 0.71, 0.72]))
    out = r("task")
    assert out.stopped_because == "plateau" and len(out.attempts) == 2


def test_regression_keeps_the_better_output():
    r = BoundedReflexion(generate=gen, judge=climbing([0.9, 0.2]))
    out = r("task")
    assert out.stopped_because == "regression"
    assert out.score == 0.9


def test_improvement_is_reported():
    r = BoundedReflexion(generate=gen, judge=climbing([0.2, 0.9]))
    assert r("task").improvement == pytest.approx(0.7)


def test_constraints_are_passed_forward():
    seen = []

    def watching_gen(task, constraints):
        seen.append(list(constraints))
        return "out"

    BoundedReflexion(generate=watching_gen, judge=climbing([0.2, 0.9]))("task")
    assert seen[0] == [] and seen[1] == ["Do better."]


def test_on_attempt_callback_fires():
    seen = []
    r = BoundedReflexion(generate=gen, judge=climbing([0.2, 0.9]),
                         on_attempt=seen.append)
    out = r("task")
    # 0.2 -> 0.9 improves, 0.9 -> 0.9 plateaus and stops
    assert len(seen) == 3 and out.stopped_because == "plateau"
    assert all(a.ms >= 0 for a in seen)


def test_timeout_stops_the_loop():
    r = BoundedReflexion(generate=gen, judge=climbing([0.1, 0.4, 0.8]), max_seconds=-1)
    assert r("task").stopped_because == "timeout"


def test_as_node_returns_state_dict():
    node = BoundedReflexion(generate=gen, judge=climbing([0.2, 0.9])).as_node()
    state = node({"task": "explain backpressure", "unrelated": 1})
    assert state["unrelated"] == 1
    assert state["output"].startswith("answer")
    assert state["reflexion"]["attempts"] == 3
    assert state["reflexion"]["stopped_because"] == "plateau"


def test_docstring_example_holds():
    refine = BoundedReflexion(
        generate=lambda task, cons: "42" if cons else "forty-two",
        judge=lambda task, out: (1.0, []) if out == "42" else (0.4, ["Use digits."]),
    )
    assert refine("What is six times seven?").best == "42"
