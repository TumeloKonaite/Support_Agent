import json
import tempfile
import unittest
from pathlib import Path

from src.app.domain.support.models import ConversationTurn
from src.app.infrastructure.storage.file_conversation_store import (
    FileConversationStore,
)


class FileConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "conversations"
        self.store = FileConversationStore(self.storage_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_returns_empty_list_when_file_missing(self) -> None:
        self.assertEqual(self.store.load("missing-session"), [])

    def test_load_returns_existing_conversation(self) -> None:
        conversation_path = self.storage_dir / "session-1.json"
        conversation_path.write_text(
            json.dumps(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            ),
            encoding="utf-8",
        )

        loaded = self.store.load("session-1")

        self.assertEqual(
            loaded,
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(role="assistant", content="hi there"),
            ],
        )

    def test_load_preserves_message_ordering_and_content(self) -> None:
        conversation_path = self.storage_dir / "session-order.json"
        conversation_path.write_text(
            json.dumps(
                [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ]
            ),
            encoding="utf-8",
        )

        loaded = self.store.load("session-order")

        self.assertEqual(
            loaded,
            [
                ConversationTurn(role="system", content="system prompt"),
                ConversationTurn(role="user", content="first"),
                ConversationTurn(role="assistant", content="second"),
                ConversationTurn(role="user", content="third"),
            ],
        )

    def test_load_returns_empty_list_when_file_is_malformed(self) -> None:
        conversation_path = self.storage_dir / "bad-session.json"
        conversation_path.write_text("{not-json", encoding="utf-8")

        self.assertEqual(self.store.load("bad-session"), [])

    def test_load_skips_malformed_messages(self) -> None:
        conversation_path = self.storage_dir / "mixed-session.json"
        conversation_path.write_text(
            json.dumps(
                [
                    {"role": "user", "content": "keep me"},
                    {"role": "assistant"},
                    {"role": 123, "content": "bad role"},
                    "bad item",
                    {"role": "assistant", "content": "also keep me"},
                ]
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            self.store.load("mixed-session"),
            [
                ConversationTurn(role="user", content="keep me"),
                ConversationTurn(role="assistant", content="also keep me"),
            ],
        )

    def test_save_persists_conversation_as_json(self) -> None:
        self.store.save(
            "session-2",
            [
                ConversationTurn(role="user", content="question"),
                ConversationTurn(role="assistant", content="answer"),
            ],
        )

        conversation_path = self.storage_dir / "session-2.json"
        self.assertTrue(conversation_path.exists())
        self.assertEqual(
            json.loads(conversation_path.read_text(encoding="utf-8")),
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )
