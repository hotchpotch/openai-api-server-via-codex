from __future__ import annotations

from openai_api_server_via_codex.compat import (
    ChatCompletionStore,
    ResponseStore,
    normalize_response_input,
)


def test_response_store_evicts_oldest_entry_when_max_entries_is_reached() -> None:
    store = ResponseStore(max_entries=2)

    for index in range(3):
        store.remember(
            f"resp_{index}",
            effective_input=[{"role": "user", "content": f"input {index}"}],
            response={
                "id": f"resp_{index}",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f"output {index}"}
                        ],
                    }
                ],
            },
        )

    assert store.get("resp_0") is None
    assert store.get("resp_1") is not None
    assert store.get("resp_2") is not None
    assert store.size == 2


def test_response_store_max_entries_zero_disables_storage() -> None:
    store = ResponseStore(max_entries=0)

    store.remember(
        "resp_1",
        effective_input=[{"role": "user", "content": "input"}],
        response={"id": "resp_1", "output": []},
    )

    assert store.get("resp_1") is None
    assert store.size == 0


def test_normalize_response_input_preserves_reasoning_summary_with_encrypted_content() -> None:
    normalized = normalize_response_input(
        [
            {
                "type": "reasoning",
                "encrypted_content": "ciphertext-abc",
                "summary": [{"type": "summary_text", "text": "Plan steps."}],
            }
        ]
    )

    assert normalized == [
        {
            "type": "reasoning",
            "encrypted_content": "ciphertext-abc",
            "summary": [{"type": "summary_text", "text": "Plan steps."}],
        }
    ]


def test_normalize_response_input_defaults_reasoning_summary_to_empty_list() -> None:
    normalized = normalize_response_input(
        [
            {
                "type": "reasoning",
                "encrypted_content": "ciphertext-xyz",
            }
        ]
    )

    assert normalized == [
        {
            "type": "reasoning",
            "encrypted_content": "ciphertext-xyz",
            "summary": [],
        }
    ]


def test_normalize_response_input_drops_reasoning_without_encrypted_content() -> None:
    normalized = normalize_response_input(
        [
            {
                "type": "reasoning",
                "summary": [],
            },
            {"role": "user", "content": "hi"},
        ]
    )

    assert normalized == [{"role": "user", "content": "hi"}]


def test_chat_completion_store_evicts_oldest_entry_when_max_entries_is_reached() -> None:
    store = ChatCompletionStore(max_entries=2)

    for index in range(3):
        store.remember(
            f"chatcmpl_{index}",
            completion={
                "id": f"chatcmpl_{index}",
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"reply {index}"},
                    }
                ],
            },
        )

    items, has_more = store.list()

    assert [item["id"] for item in items] == ["chatcmpl_1", "chatcmpl_2"]
    assert has_more is False
    assert store.get("chatcmpl_0") is None
    assert store.get("chatcmpl_1") is not None
    assert store.size == 2
