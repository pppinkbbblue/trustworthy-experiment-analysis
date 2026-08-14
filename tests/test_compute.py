import math

from cve.compute import Arm, compare_arms, decision


def test_rate():
    a = Arm("t", users=1000, conversions=250)
    assert abs(a.rate - 0.25) < 1e-9


def test_no_effect_is_symmetric():
    c = Arm("control", 100_000, 10_000)
    t = Arm("T1", 100_000, 10_000)
    r = compare_arms(c, t, "conversion:T1")
    assert abs(r.absolute_lift) < 1e-9
    assert abs(r.prob_treatment_better - 0.5) < 1e-6
    assert r.p_value > 0.99


def test_clear_positive_effect():
    c = Arm("control", 100_000, 10_000)      # 10%
    t = Arm("T1", 100_000, 13_000)           # 13%
    r = compare_arms(c, t, "conversion:T1")
    assert r.absolute_lift > 0
    assert r.relative_lift_pct > 25
    assert r.prob_treatment_better > 0.99
    assert r.p_value < 0.001
    assert decision(r) == "positive"


def test_clear_negative_effect():
    c = Arm("control", 100_000, 13_000)
    t = Arm("T1", 100_000, 10_000)
    r = compare_arms(c, t, "conversion:T1")
    assert r.absolute_lift < 0
    assert decision(r) == "negative"


def test_facts_keys():
    c = Arm("control", 5000, 500)
    t = Arm("T1", 5000, 600)
    r = compare_arms(c, t, "conversion:T1")
    facts = r.facts()
    for k in ["treatment_rate", "control_rate", "relative_lift_pct",
              "prob_treatment_better", "p_value"]:
        assert k in facts
