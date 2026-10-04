"""MCP tool parity with the Hermes provider surface.

The persona tools (promote/demote/list/reinforce), the sync tools
(push/pull/status) and ``mnemosyne_triple_end`` were Hermes-provider-only;
they are now served by the MCP dispatcher too. These tests pin three
contracts:

1. The persona and triple_end handlers route on the per-call ``bank`` and
   the advertised MCP schema declares it (on the MCP copy -- the canonical
   schemas stay bank-free because the Hermes provider pins them verbatim).
2. The sync tools act on the configured sync state (no bank) and are
   fail-closed: undiscoverable and undispachable until
   ``MNEMOSYNE_MCP_SYNC_TOOLS`` opts the operator in.
3. The response shapes match the Hermes provider handlers, so a client sees
   the same fields on either surface.

Run with: pytest tests/test_mcp_tool_parity.py -v
"""

import pytest

from mnemosyne.mcp_tools import (
    _SYNC_TOOL_NAMES,
    _TOOL_HANDLERS,
    _sync_tools_enabled,
    get_tool_definitions,
    handle_tool_call,
)
from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS, BANK_EXEMPT_TOOLS

PERSONA_TOOLS = (
    "mnemosyne_persona_promote",
    "mnemosyne_persona_demote",
    "mnemosyne_persona_list",
    "mnemosyne_persona_reinforce",
)


@pytest.fixture(autouse=True)
def _sync_gate_off(monkeypatch):
    """The sync gate defaults to off; keep tests independent of the host env."""
    monkeypatch.delenv("MNEMOSYNE_MCP_SYNC_TOOLS", raising=False)


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Point every Mnemosyne path at an empty per-test directory."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    return tmp_path


class TestPersonaToolsOverMcp:
    def test_persona_tools_are_advertised_with_bank(self):
        defs = {t["name"]: t for t in get_tool_definitions()}
        for name in PERSONA_TOOLS:
            assert name in defs
            assert "bank" in defs[name]["input_schema"]["properties"]

    def test_canonical_persona_schemas_stay_bank_free(self):
        """The Hermes provider pins the canonical persona schemas verbatim."""
        assert set(PERSONA_TOOLS) <= BANK_EXEMPT_TOOLS
        for schema in ALL_TOOL_SCHEMAS:
            if schema["name"] in PERSONA_TOOLS:
                assert "bank" not in schema["parameters"]["properties"]

    def test_promote_happy_path(self, fresh_home):
        stored = handle_tool_call("mnemosyne_remember", {
            "content": "Always answer with citations",
            "importance": 0.9,
        })
        assert stored["status"] == "stored"

        result = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": stored["memory_id"],
            "tier": "permanent",
            "reason": "core rule",
        })

        assert result["status"] == "ok"
        assert isinstance(result["persona_id"], int)
        assert result["tier"] == "permanent"
        assert result["topic"] == "general"

    def test_promote_rejects_invalid_tier(self, fresh_home):
        result = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": "whatever", "tier": "eternal",
        })
        assert result["status"] == "error"
        assert "Invalid tier" in result["error"]

    def test_promote_reports_missing_source_memory(self, fresh_home):
        result = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": "does-not-exist",
        })
        assert result == {
            "status": "error",
            "error": "Source memory 'does-not-exist' not found.",
        }

    def test_demote_moves_persona_to_preferences(self, fresh_home):
        stored = handle_tool_call("mnemosyne_remember", {"content": "prefers short answers"})
        promoted = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": stored["memory_id"], "tier": "long_term",
        })
        persona_id = promoted["persona_id"]

        result = handle_tool_call("mnemosyne_persona_demote", {
            "persona_id": persona_id, "reason": "one-off",
        })
        assert result == {
            "status": "ok",
            "persona_id": persona_id,
            "demoted_to": "memoria_preferences",
        }

        listed = handle_tool_call("mnemosyne_persona_list", {})
        assert listed["count"] == 0

    def test_demote_reports_unknown_persona(self, fresh_home):
        result = handle_tool_call("mnemosyne_persona_demote", {"persona_id": 424242})
        assert result == {"status": "error", "error": "Persona id 424242 not found."}

    def test_reinforce_bumps_count_and_reports_unknown(self, fresh_home):
        stored = handle_tool_call("mnemosyne_remember", {"content": "uses metric units"})
        promoted = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": stored["memory_id"],
        })
        persona_id = promoted["persona_id"]

        assert handle_tool_call("mnemosyne_persona_reinforce", {
            "persona_id": persona_id,
        }) == {"status": "ok", "persona_id": persona_id, "reinforced": True}
        assert handle_tool_call("mnemosyne_persona_reinforce", {
            "persona_id": 987654,
        }) == {"status": "error", "error": "Persona id 987654 not found."}

        listed = handle_tool_call("mnemosyne_persona_list", {})
        assert listed["personas"][0]["reinforcement_count"] == 1
        assert listed["personas"][0]["last_reinforced_at"]

    def test_list_orders_permanent_first_and_filters(self, fresh_home):
        ids = []
        for content in ("rule A", "rule B", "rule C"):
            stored = handle_tool_call("mnemosyne_remember", {"content": content})
            ids.append(stored["memory_id"])
        handle_tool_call("mnemosyne_persona_promote", {"memory_id": ids[0], "tier": "long_term"})
        handle_tool_call("mnemosyne_persona_promote", {"memory_id": ids[1], "tier": "permanent"})
        handle_tool_call("mnemosyne_persona_promote", {"memory_id": ids[2], "tier": "working"})

        listed = handle_tool_call("mnemosyne_persona_list", {})
        assert [p["tier"] for p in listed["personas"]] == [
            "permanent", "long_term", "working",
        ]

        by_tier = handle_tool_call("mnemosyne_persona_list", {"tier": "working"})
        assert [p["tier"] for p in by_tier["personas"]] == ["working"]
        assert by_tier["personas"][0]["content"] == "rule C"

        # Declared-but-blank optionals mean "omitted" (#555-style contract).
        blank = handle_tool_call("mnemosyne_persona_list", {"tier": "", "topic": "  "})
        assert blank["count"] == 3

    def test_persona_stores_are_bank_scoped(self, fresh_home):
        stored = handle_tool_call("mnemosyne_remember", {
            "content": "tenant A rule", "bank": "tenant_a",
        })
        promoted = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": stored["memory_id"], "bank": "tenant_a",
        })
        assert promoted["status"] == "ok"

        assert handle_tool_call("mnemosyne_persona_list", {"bank": "tenant_a"})["count"] == 1
        assert handle_tool_call("mnemosyne_persona_list", {"bank": "tenant_b"})["count"] == 0


class TestTripleEndOverMcp:
    def test_triple_end_is_advertised_with_bank(self):
        defs = {t["name"]: t for t in get_tool_definitions()}
        assert "mnemosyne_triple_end" in defs
        assert "bank" in defs["mnemosyne_triple_end"]["input_schema"]["properties"]

    def test_triple_end_closes_open_triples(self, fresh_home):
        added = handle_tool_call("mnemosyne_triple_add", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
        })
        assert added["status"] == "added"

        result = handle_tool_call("mnemosyne_triple_end", {
            "subject": "Alice", "predicate": "works_with",
        })
        assert result == {"status": "ended", "count": 1}

        query = handle_tool_call("mnemosyne_triple_query", {
            "subject": "Alice", "predicate": "works_with",
        })
        assert query["results_count"] == 0

    def test_triple_end_targets_one_object_when_given(self, fresh_home):
        handle_tool_call("mnemosyne_triple_add", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
        })

        # A non-matching object closes nothing and leaves the triple open.
        miss = handle_tool_call("mnemosyne_triple_end", {
            "subject": "Alice", "predicate": "works_with", "object": "Carol",
        })
        assert miss == {"status": "ended", "count": 0}
        query = handle_tool_call("mnemosyne_triple_query", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
        })
        assert query["results_count"] == 1

        result = handle_tool_call("mnemosyne_triple_end", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
        })
        assert result == {"status": "ended", "count": 1}
        query = handle_tool_call("mnemosyne_triple_query", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
        })
        assert query["results_count"] == 0

    def test_triple_end_requires_subject_and_predicate(self, fresh_home):
        assert handle_tool_call("mnemosyne_triple_end", {"subject": "Alice"}) == {
            "error": "subject and predicate are required",
        }
        assert handle_tool_call("mnemosyne_triple_end", {}) == {
            "error": "subject and predicate are required",
        }

    def test_triple_end_routes_to_the_requested_bank(self, fresh_home):
        handle_tool_call("mnemosyne_triple_add", {
            "subject": "Alice", "predicate": "works_with", "object": "Bob",
            "bank": "tenant_a",
        })

        result = handle_tool_call("mnemosyne_triple_end", {
            "subject": "Alice", "predicate": "works_with", "bank": "tenant_a",
        })
        assert result == {"status": "ended", "count": 1}

        handle_tool_call("mnemosyne_triple_add", {
            "subject": "Alice", "predicate": "mentors", "object": "Bob",
            "bank": "tenant_b",
        })
        other = handle_tool_call("mnemosyne_triple_end", {
            "subject": "Alice", "predicate": "works_with", "bank": "tenant_b",
        })
        assert other == {"status": "ended", "count": 0}


class TestSyncGate:
    def test_sync_tools_are_hidden_and_refused_by_default(self):
        names = {t["name"] for t in get_tool_definitions()}
        assert names.isdisjoint(_SYNC_TOOL_NAMES)

        for name in sorted(_SYNC_TOOL_NAMES):
            with pytest.raises(ValueError, match="MNEMOSYNE_MCP_SYNC_TOOLS"):
                handle_tool_call(name, {})

    def test_sync_tools_stay_in_the_handler_registry(self):
        assert _SYNC_TOOL_NAMES <= set(_TOOL_HANDLERS)

    @pytest.mark.parametrize("value", ["1", "true", "Yes", "ON", "  on  "])
    def test_gate_accepts_opt_in_spellings(self, monkeypatch, value):
        monkeypatch.setenv("MNEMOSYNE_MCP_SYNC_TOOLS", value)
        assert _sync_tools_enabled() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "off", "no", "enabled!"])
    def test_gate_rejects_other_spellings(self, monkeypatch, value):
        monkeypatch.setenv("MNEMOSYNE_MCP_SYNC_TOOLS", value)
        assert _sync_tools_enabled() is False

    def test_enabled_gate_advertises_sync_schemas_without_bank(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_SYNC_TOOLS", "1")
        defs = {t["name"]: t for t in get_tool_definitions()}
        assert _SYNC_TOOL_NAMES <= set(defs)
        for name in _SYNC_TOOL_NAMES:
            assert "bank" not in defs[name]["input_schema"]["properties"]
            # The canonical sync schema carries the gate note so clients see
            # why the tool may be absent from an unconfigured server.
            assert "MNEMOSYNE_MCP_SYNC_TOOLS" in defs[name]["description"]

    def test_enabled_gate_surface_is_constructible(self, monkeypatch):
        mcp_types = pytest.importorskip("mcp.types")
        monkeypatch.setenv("MNEMOSYNE_MCP_SYNC_TOOLS", "1")
        for t in get_tool_definitions():
            mcp_types.Tool(**t)


class _FakeSyncEngine:
    """Records constructor and sync_with calls; never touches the network."""

    instances = []

    def __init__(self, beam_instance, encryption=None, **kwargs):
        self.beam_instance = beam_instance
        self.encryption = encryption
        self.kwargs = kwargs
        self.device_id = "fake-device"
        self.conn = beam_instance.conn
        self.calls = []
        type(self).instances.append(self)

    def sync_with(self, remote_url, mode="bidirectional", api_key=None, encryption_key=None):
        self.calls.append({
            "remote": remote_url, "mode": mode, "api_key": api_key,
        })
        if mode == "push":
            return {
                "remote": remote_url, "mode": mode, "errors": [],
                "push": {"accepted": 3, "duplicates": 1, "conflicts": 0},
                "pull": None,
            }
        return {
            "remote": remote_url, "mode": mode, "errors": [],
            "push": None,
            "pull": {"accepted": 2, "duplicates": 0, "conflicts": 1},
        }

    def _meta_get(self, key, default=None):
        return ""


@pytest.fixture
def sync_surface(tmp_path, monkeypatch):
    """A dedicated surface DB wired in as the MCP shared surface."""
    from mnemosyne.core.beam import BeamMemory

    surface = BeamMemory(
        session_id="mcp_shared_surface", db_path=str(tmp_path / "surface.db")
    )
    monkeypatch.setattr(
        "mnemosyne.mcp_tools._create_surface_instance", lambda: surface
    )
    return surface


@pytest.fixture
def fake_sync_engine(monkeypatch):
    """Swap in the recording fake engine with a fresh instance list."""
    monkeypatch.setattr("mnemosyne.core.sync.SyncEngine", _FakeSyncEngine)
    monkeypatch.setattr(_FakeSyncEngine, "instances", [])


class TestSyncToolsWhenEnabled:
    @pytest.fixture(autouse=True)
    def _gate_on(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_MCP_SYNC_TOOLS", "1")

    def test_push_without_remote_reports_configuration_error(self, sync_surface):
        assert handle_tool_call("mnemosyne_sync_push", {}) == {
            "status": "error",
            "error": "No remote configured. Set MNEMOSYNE_SYNC_REMOTE env var.",
        }

    def test_pull_without_remote_reports_configuration_error(self, sync_surface):
        assert handle_tool_call("mnemosyne_sync_pull", {}) == {
            "status": "error",
            "error": "No remote configured. Set MNEMOSYNE_SYNC_REMOTE env var.",
        }

    def test_status_reports_unconfigured_surface(self, sync_surface):
        result = handle_tool_call("mnemosyne_sync_status", {})
        assert result["status"] == "ok"
        assert result["device_id"]
        assert result["remote"] == "(unconfigured)"
        assert result["encryption"] == "disabled"
        assert result["mode"] == "bidirectional"
        assert result["local_events"] >= 0
        assert result["last_cursor"] == "none"

    def test_push_and_pull_forward_remote_mode_and_token(
        self, sync_surface, monkeypatch, fake_sync_engine
    ):
        monkeypatch.setenv("MNEMOSYNE_SYNC_REMOTE", "https://sync.example.com:8443")
        monkeypatch.setenv("MNEMOSYNE_SYNC_TOKEN", "secret-token")

        pushed = handle_tool_call("mnemosyne_sync_push", {})
        assert pushed == {
            "status": "ok", "pushed": 3, "duplicates": 1,
            "conflicts": 0, "next_cursor": "",
        }
        pulled = handle_tool_call("mnemosyne_sync_pull", {})
        assert pulled == {
            "status": "ok", "pulled": 2, "duplicates": 0,
            "conflicts": 1, "next_cursor": "",
        }

        # One engine per call, each with the configured remote, the requested
        # mode and the bearer token from the environment.
        assert len(_FakeSyncEngine.instances) == 2
        assert _FakeSyncEngine.instances[0].calls == [{
            "remote": "https://sync.example.com:8443",
            "mode": "push",
            "api_key": "secret-token",
        }]
        assert _FakeSyncEngine.instances[1].calls == [{
            "remote": "https://sync.example.com:8443",
            "mode": "pull",
            "api_key": "secret-token",
        }]

    def test_engine_is_surface_only_over_the_shared_surface(
        self, sync_surface, monkeypatch, fake_sync_engine
    ):
        monkeypatch.setenv("MNEMOSYNE_SYNC_REMOTE", "https://sync.example.com")

        handle_tool_call("mnemosyne_sync_push", {})

        fake = _FakeSyncEngine.instances[-1]
        assert fake.kwargs["surface_only"] is True
        assert fake.kwargs["surface_id"] == "shared-surface-v1"
        assert fake.kwargs["initialize_surface"] is True
        assert fake.beam_instance is sync_surface

    def test_encryption_key_reaches_the_engine(
        self, sync_surface, monkeypatch, fake_sync_engine
    ):
        import base64

        monkeypatch.setenv("MNEMOSYNE_SYNC_REMOTE", "https://sync.example.com")
        monkeypatch.setenv("MNEMOSYNE_SYNC_ENCRYPT", "1")
        # from_config expects a base64 key; a malformed one raises and the
        # handler reports the engine as unavailable instead.
        key = base64.urlsafe_b64encode(b"0123456789abcdef" * 2).decode()
        monkeypatch.setenv("MNEMOSYNE_SYNC_KEY", key)

        handle_tool_call("mnemosyne_sync_push", {})

        fake = _FakeSyncEngine.instances[-1]
        assert fake.encryption is not None

    def test_foreign_rows_refuse_engine_construction(self, sync_surface, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_SYNC_REMOTE", "https://sync.example.com")
        sync_surface.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, scope) "
            "VALUES ('foreign-1', 'x', 'conversation', '2026-01-01', 'other-session', 'session')"
        )
        sync_surface.conn.commit()

        result = handle_tool_call("mnemosyne_sync_push", {})
        assert result["status"] == "error"
        assert "Sync adapter unavailable" in result["error"]
        assert "dedicated global shared surface" in result["error"]
