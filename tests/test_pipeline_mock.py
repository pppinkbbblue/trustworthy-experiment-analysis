from cve.data import generate_dataset
from cve.llm import get_client
from cve.pipeline import run_cve, run_llm_only


def test_end_to_end_mock_runs():
    client = get_client("mock")
    exp = generate_dataset(n=1, seed=1)[0]
    c = run_cve(client, exp)
    assert c.condition == "cve"
    assert c.grounding is not None
    # CVE narrative must not contain ungrounded numbers.
    assert c.grounding.hallucination_rate == 0.0
    # Deterministic decisions exist for every treatment.
    assert set(c.decisions) == {t.name for t in exp.treatments}

    l = run_llm_only(client, exp)
    assert l.condition == "llm_only"
