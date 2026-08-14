from cve.validate import extract_numbers, validate_grounding


def test_extract_numbers():
    nums = extract_numbers("Lift was 12.5% (p=0.03), 1000 users.")
    assert 12.5 in nums and 0.03 in nums and 1000.0 in nums


def test_grounded_numbers_pass():
    facts = {"m:relative_lift_pct": 12.5, "m:p_value": 0.03}
    rep = validate_grounding("The relative lift was 12.5% with p=0.03.", facts)
    assert rep.hallucination_rate == 0.0
    assert rep.traceability == 1.0


def test_ungrounded_number_flagged():
    facts = {"m:relative_lift_pct": 12.5}
    # 99.9 is not among the computed facts -> should be flagged.
    rep = validate_grounding("Lift 12.5%, but revenue rose 99.9%.", facts)
    assert 99.9 in rep.ungrounded
    assert rep.hallucination_rate > 0.0


def test_percentage_form_of_rate_is_grounded():
    facts = {"m:treatment_rate": 0.234}
    rep = validate_grounding("The treatment converted at 23.4%.", facts)
    assert rep.hallucination_rate == 0.0
