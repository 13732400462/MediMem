from mem_ehr_agent.extractive_timeline import build_extractive_timeline


def turns():
    return [
        {
            "session": f"s{session}",
            "session_date": f"2026-01-{session + 1:02d}",
            "speaker": "user",
            "text": f"session {session} " + ("detail " * 100),
            "evidence_refs": [f"ref-{session}"],
        }
        for session in range(12)
    ]


def test_extractive_timeline_is_deterministic_and_bounded():
    first, first_diagnostics = build_extractive_timeline(turns(), max_chars=6500)
    second, second_diagnostics = build_extractive_timeline(turns(), max_chars=6500)
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert len(first) <= 6500


def test_extractive_timeline_preserves_session_and_temporal_coverage():
    memory, diagnostics = build_extractive_timeline(turns(), max_chars=6500)
    assert "session=s0" in memory
    assert "session=s11" in memory
    assert diagnostics["retained_session_count"] == 12


def test_extractive_timeline_handles_empty_and_oversized_turns():
    empty, empty_diagnostics = build_extractive_timeline([], max_chars=6500)
    assert "No timeline facts stored" in empty
    assert empty_diagnostics["retained_record_count"] == 0
    oversized, diagnostics = build_extractive_timeline(
        [{"session": "s1", "text": "x" * 20_000}],
        max_chars=6500,
    )
    assert len(oversized) <= 6500
    assert diagnostics["records_truncated"] == 1


def test_extractive_timeline_uniformly_samples_too_many_sessions():
    many = [{"session": f"s{index}", "text": f"fact {index}"} for index in range(200)]
    memory, diagnostics = build_extractive_timeline(many, max_chars=1000)
    assert len(memory) <= 1000
    assert diagnostics["sessions_sampled"] is True
    assert "session=s0" in memory
    assert "session=s199" in memory
