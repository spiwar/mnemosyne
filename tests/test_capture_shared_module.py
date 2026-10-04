"""Golden check: the shared turn-capture module and the Hermes provider write
identical memory kwargs, pinned to exact expected values."""

from __future__ import annotations

import pytest

import hermes_memory_provider
from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core.prefetch import (
    _sync_turn_assistant_limit,
    _sync_turn_user_limit,
    capture_turn,
)


class FakeLedger:
    enabled = True

    def __init__(self):
        self.begins = []
        self.captures = []

    def begin(self, session_id, messages):
        self.begins.append((session_id, messages))
        return f"ticket-{len(self.begins)}"

    def capture(self, session_id, ticket, beam, raw_content, **kwargs):
        self.captures.append((session_id, ticket, beam, raw_content, kwargs))
        return f"cap-{len(self.captures)}"


class FakeBeam:
    def __init__(self):
        self.remember_calls = []

    def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        return f"mem-{len(self.remember_calls)}"


USER_TEXT = "Remember that the deploy window moved to Fridays"
ASSISTANT_TEXT = "The deploy window now lands on Friday afternoons"

EXPECTED_USER_KWARGS = {
    "content": f"[USER] {USER_TEXT}",
    "source": "conversation",
    "importance": 0.5,
    "scope": "session",
    "extract_entities": True,
    "_write_policy_content": USER_TEXT,
}
EXPECTED_ASSISTANT_KWARGS = {
    "content": f"[ASSISTANT] {ASSISTANT_TEXT}",
    "source": "conversation",
    "importance": 0.15,
    "scope": "session",
    "extract_entities": True,
    "_write_policy_content": ASSISTANT_TEXT,
}


@pytest.fixture(autouse=True)
def _default_limits(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", raising=False)
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", raising=False)


def _provider(beam, ledger=None):
    provider = object.__new__(MnemosyneMemoryProvider)
    provider._beam = beam
    provider._agent_context = "primary"
    provider._skip_contexts = set()
    provider._sync_roles = {"user", "assistant"}
    provider._default_scope = "session"
    provider._turn_count = 0
    provider._auto_sleep_enabled = False
    if ledger is not None:
        provider._verbatim_ledger = ledger
    return provider


def test_capture_turn_golden_kwargs():
    beam = FakeBeam()
    returned = capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, policy=None)
    assert beam.remember_calls == [EXPECTED_USER_KWARGS, EXPECTED_ASSISTANT_KWARGS]
    assert returned == "mem-1"


def test_provider_sync_turn_matches_shared_module():
    beam = FakeBeam()
    ledger = FakeLedger()
    provider = _provider(beam, ledger)
    provider._active_session_id = "s1"  # so sync_turn mints a ledger ticket
    provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="s1")
    assert provider._turn_count == 1
    assert ledger.begins == [("s1", None)]
    (ticket,) = {c[1] for c in ledger.captures}
    assert ledger.captures == [
        ("s1", ticket, beam, USER_TEXT, EXPECTED_USER_KWARGS),
        ("s1", ticket, beam, ASSISTANT_TEXT, EXPECTED_ASSISTANT_KWARGS),
    ]
    # The ledger capture lambda replaces beam.remember entirely.
    assert beam.remember_calls == []

    direct_beam = FakeBeam()
    direct_ledger = FakeLedger()
    returned = capture_turn(
        direct_beam, USER_TEXT, ASSISTANT_TEXT,
        ledger=direct_ledger, ledger_session_id="s1", ticket="t",
        policy=None,
    )
    assert returned == "cap-1"
    assert direct_ledger.captures == [
        ("s1", "t", direct_beam, USER_TEXT, EXPECTED_USER_KWARGS),
        ("s1", "t", direct_beam, ASSISTANT_TEXT, EXPECTED_ASSISTANT_KWARGS),
    ]


def test_provider_no_ledger_writes_through_beam():
    beam = FakeBeam()
    provider = _provider(beam)
    provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="s1")
    assert beam.remember_calls == [EXPECTED_USER_KWARGS, EXPECTED_ASSISTANT_KWARGS]


def test_role_gating():
    beam = FakeBeam()
    returned = capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, roles=("user",), policy=None)
    assert beam.remember_calls == [EXPECTED_USER_KWARGS]
    assert returned == "mem-1"

    beam = FakeBeam()
    returned = capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, roles=("assistant",), policy=None)
    assert beam.remember_calls == [EXPECTED_ASSISTANT_KWARGS]
    assert returned is None

    beam = FakeBeam()
    assert capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, roles=(), policy=None) is None
    assert beam.remember_calls == []


def test_length_gates():
    beam = FakeBeam()
    # Exactly at each gate: user > 5, assistant > 10.
    assert capture_turn(beam, "12345", "1234567890", policy=None) is None
    assert beam.remember_calls == []

    capture_turn(beam, "123456", "12345678901", policy=None)
    assert beam.remember_calls == [
        {**EXPECTED_USER_KWARGS, "content": "[USER] 123456",
         "_write_policy_content": "123456"},
        {**EXPECTED_ASSISTANT_KWARGS, "content": "[ASSISTANT] 12345678901",
         "_write_policy_content": "12345678901"},
    ]

    beam = FakeBeam()
    capture_turn(beam, "", "", policy=None)
    assert beam.remember_calls == []


def test_default_scope_override():
    beam = FakeBeam()
    capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, default_scope="global", policy=None)
    assert beam.remember_calls == [
        {**EXPECTED_USER_KWARGS, "scope": "global"},
        {**EXPECTED_ASSISTANT_KWARGS, "scope": "global"},
    ]


def test_limits_truncate_raw_not_prefixed(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "10")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "12")
    beam = FakeBeam()
    capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, policy=None)
    assert beam.remember_calls == [
        {**EXPECTED_USER_KWARGS, "content": f"[USER] {USER_TEXT[:10]}"},
        {**EXPECTED_ASSISTANT_KWARGS, "content": f"[ASSISTANT] {ASSISTANT_TEXT[:12]}"},
    ]
    # The policy content stays the raw, untruncated turn text.
    assert beam.remember_calls[0]["_write_policy_content"] == USER_TEXT
    assert beam.remember_calls[1]["_write_policy_content"] == ASSISTANT_TEXT


def test_limit_zero_means_untruncated(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "0")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "0")
    long_user = "u" * 1200
    long_assistant = "a" * 1200
    beam = FakeBeam()
    capture_turn(beam, long_user, long_assistant, policy=None)
    assert beam.remember_calls == [
        {**EXPECTED_USER_KWARGS, "content": f"[USER] {long_user}",
         "_write_policy_content": long_user},
        {**EXPECTED_ASSISTANT_KWARGS, "content": f"[ASSISTANT] {long_assistant}",
         "_write_policy_content": long_assistant},
    ]


def test_default_limits_and_negative_fallback(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", raising=False)
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", raising=False)
    assert _sync_turn_user_limit() == 500
    assert _sync_turn_assistant_limit() == 800
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "-10")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "bogus")
    assert _sync_turn_user_limit() == 0
    assert _sync_turn_assistant_limit() == 800


def test_policy_active_and_lock_held_during_writes():
    from mnemosyne.core.filters import active_write_policy, resolve_write_policy

    policy = resolve_write_policy()
    events = []

    class Recorder:
        def __enter__(self):
            events.append("lock")
            return self

        def __exit__(self, *exc):
            events.append("unlock")
            return False

    class RecordingBeam(FakeBeam):
        def remember(self, **kwargs):
            events.append("write")
            assert active_write_policy() is policy
            return super().remember(**kwargs)

    beam = RecordingBeam()
    capture_turn(beam, USER_TEXT, ASSISTANT_TEXT, policy=policy, lock=Recorder())
    assert events == ["lock", "write", "write", "unlock"]


def test_moved_names_reexported_from_provider():
    from mnemosyne.core import prefetch

    assert hermes_memory_provider.capture_turn is prefetch.capture_turn
    assert hermes_memory_provider._sync_turn_user_limit is prefetch._sync_turn_user_limit
    assert hermes_memory_provider._sync_turn_assistant_limit is prefetch._sync_turn_assistant_limit
    assert hermes_memory_provider._sync_turn_user_limit() == 500
    assert hermes_memory_provider._sync_turn_assistant_limit() == 800
