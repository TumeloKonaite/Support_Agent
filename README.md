# Support Agent Starter

[Python](https://www.python.org/)
[FastAPI](https://fastapi.tiangolo.com/)
[OpenAI](https://platform.openai.com/docs)
[pytest](https://docs.pytest.org/)

A small FastAPI customer-support assistant starter. The app loads business-managed content from `data/`, builds a support prompt from that content plus conversation history, retrieves relevant indexed knowledge, calls OpenAI for chat responses, and stores transcripts on disk.

## Current Status

The project is in a solid starter state rather than a finished product.

Implemented today:

- FastAPI app with `GET /`, `GET /health`, `POST /chat`, and `POST /chat/stream`
- OpenAI-backed chat and streaming chat responses
- File-backed conversation history keyed by `session_id`
- JSON-backed business profile and support knowledge loaders
- Local retrieval pipeline that indexes `data/` into a JSON vector store
- Request-time retrieval that injects matched business context into prompts
- Unit and route tests covering loaders, prompt building, retrieval, storage, service behavior, and API routes

Not yet exposed or still intentionally simple:

- No `tenant_id` in the public chat API, even though the loaders support tenant-specific files internally
- No database or hosted vector store; retrieval persists to a local JSON file
- No auth, admin UI, background jobs, or production deployment setup
- Chat responses do not return the generated `session_id`, so clients should send their own if they want continuity

## Features

- FastAPI API for synchronous and streaming support chat
- OpenAI Responses API integration behind an isolated LLM client
- Business profile and support knowledge loaded from editable JSON files
- Retrieval-augmented prompting using a local vector index
- Disk-based conversation storage for simple local development
- Layered structure across API, domain, infrastructure, and tests

## Project Structure

```text
.
├── data/
│   ├── business_profile.json       # Business identity, contact details, tone, metadata
│   ├── knowledge.json              # Policies, FAQs, services, product knowledge
│   ├── conversations/              # Stored chat transcripts by session id
│   └── retrieval/
│       └── vector_store.json       # Generated local retrieval index
├── src/app/
│   ├── api/                        # FastAPI routes and request/response schemas
│   ├── core/                       # Settings and dependency wiring
│   ├── domain/support/             # Prompt building and support service logic
│   └── infrastructure/             # OpenAI, content loading, retrieval, storage
├── tests/                          # Unit and route tests
├── main.py                         # Simple local Python entry point
├── pyproject.toml                  # Project metadata and dependencies
└── uv.lock                         # Locked dependency versions
```

## Requirements

- Python 3.12+
- `uv`
- `OPENAI_API_KEY` for the chat endpoints

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the repo root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
APP_NAME=Support API
ENVIRONMENT=development
CONTENT_DATA_DIR=data
CONVERSATION_STORAGE_DIR=data/conversations
RETRIEVAL_EMBEDDING_PROVIDER=hashing
RETRIEVAL_EMBEDDING_MODEL=hashing-v1
RETRIEVAL_TOP_K=3
RETRIEVAL_CHUNK_SIZE=500
RETRIEVAL_CHUNK_OVERLAP=100
RETRIEVAL_VECTOR_STORE_PATH=data/retrieval/vector_store.json
```

Only `OPENAI_API_KEY` is required for chat. Retrieval defaults to the local `hashing` embedder, so you can build the knowledge index without switching to OpenAI embeddings.

If you do want OpenAI embeddings for retrieval, set:

```env
RETRIEVAL_EMBEDDING_PROVIDER=openai
```

That mode also requires `OPENAI_API_KEY`.

## Run The API

Start the FastAPI server:

```bash
uv run uvicorn src.app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your support hours?", "session_id": "demo-session"}'
```

Streaming chat request:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you help me with a refund?", "session_id": "demo-session"}'
```

## Business Content

The assistant reads its business context from the `data/` directory.

### Business Profile

Edit `data/business_profile.json` for identity and support operations:

```json
{
  "business_name": "Acme Support",
  "assistant_identity": "the Acme customer support assistant",
  "support_email": "support@acme.example",
  "support_phone": "+1-800-555-0130",
  "escalation_target": "Escalate account, refund, or policy exception requests to the human support team.",
  "support_hours": "Monday to Friday, 9:00 AM to 5:00 PM local business time.",
  "tone_guidelines": [
    "Be warm, calm, and professional.",
    "Be concise while still being helpful."
  ],
  "metadata": {
    "primary_channel": "chat",
    "industry": "ecommerce support"
  }
}
```

Required fields:

- `business_name`
- `assistant_identity`

Optional fields:

- `support_email`
- `support_phone`
- `escalation_target`
- `support_hours`
- `tone_guidelines`
- `metadata`

### Support Knowledge

Edit `data/knowledge.json` for policies, FAQs, product notes, services, handoff rules, and other support knowledge:

```json
{
  "sections": {
    "Policies": [
      "Call out uncertainty instead of inventing unsupported policies or facts.",
      "Escalate account, refund, or policy exception requests to the human support team."
    ],
    "FAQs": [
      "Customers may ask about orders, products, account access, services, and policy questions."
    ]
  }
}
```

Each key under `sections` becomes a named section in the system prompt. Each value must be a JSON array.

Useful section ideas:

- `Policies`
- `FAQs`
- `Products`
- `Services`
- `Shipping`
- `Returns`
- `Escalation Rules`
- `Brand Voice`
- `Appointments`
- `Billing`
- `Account Help`

## Retrieval Pipeline

The app does two different things with knowledge:

1. It reads structured JSON content from `data/business_profile.json` and `data/knowledge.json` directly when building prompts.
2. It separately builds a retrieval index from supported files in `data/` so the service can pull in relevant context at chat time.

Supported indexed file types:

- `.json`
- `.md`
- `.txt`

Ignored during indexing:

- `data/conversations/`

Build the local retrieval index:

```bash
uv run python -m src.app.infrastructure.retrieval.indexer
```

Test the index with a query:

```bash
uv run python -m src.app.infrastructure.retrieval.indexer --query "What are your support hours?"
```

By default, the generated vector store is written to:

```text
data/retrieval/vector_store.json
```

If the vector store is missing, chat requests still run, but retrieval will contribute no matched context.

## Tenant-Specific Files

The content loaders support tenant-specific files internally, although the current HTTP API does not expose `tenant_id`.

If domain code supplies a tenant id, files are resolved like this:

```text
data/{tenant_id}/business_profile.json
data/{tenant_id}/knowledge.json
```

If tenant-specific files are missing, the app falls back to:

```text
data/business_profile.json
data/knowledge.json
```

## Conversation Storage

Conversations are stored as JSON files in:

```text
data/conversations/
```

The filename is based on `session_id`. For example:

```text
data/conversations/demo-session.json
```

If no `session_id` is provided, the service generates one internally. The current HTTP response only returns the assistant response, so clients should provide their own `session_id` if they want to continue a conversation across requests.

## Configuration

Settings are loaded from environment variables and `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `Support API` | FastAPI application title |
| `ENVIRONMENT` | `development` | Runtime environment label |
| `API_V1_PREFIX` | empty | Reserved API prefix setting |
| `CONTENT_DATA_DIR` | `data` | Directory containing business content |
| `CONVERSATION_STORAGE_DIR` | `data/conversations` | Directory for saved transcripts |
| `OPENAI_API_KEY` | unset | Required for chat endpoints |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI model used by the LLM client |
| `RETRIEVAL_EMBEDDING_PROVIDER` | `hashing` | Retrieval embedding backend |
| `RETRIEVAL_EMBEDDING_MODEL` | `hashing-v1` | Embedding model name passed to the backend |
| `RETRIEVAL_TOP_K` | `3` | Number of chunks retrieved per request |
| `RETRIEVAL_CHUNK_SIZE` | `500` | Max chunk size used during indexing |
| `RETRIEVAL_CHUNK_OVERLAP` | `100` | Overlap between indexed chunks |
| `RETRIEVAL_VECTOR_STORE_PATH` | `data/retrieval/vector_store.json` | Local vector store output path |

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Basic app verification |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Non-streaming support chat |
| `POST` | `/chat/stream` | Streaming support chat as `text/plain` |

`POST /chat` and `POST /chat/stream` accept:

```json
{
  "message": "Hello",
  "session_id": "optional-session-id"
}
```

`POST /chat` returns:

```json
{
  "response": "Assistant response text"
}
```

## Testing

Run the full test suite:

```bash
uv run pytest
```

The test suite currently covers:

- content loaders
- prompt building
- support policies
- retrieval indexing and lookup
- conversation storage
- OpenAI client translation
- support service behavior
- API routes

## Development Notes

- Keep runtime business content in `data/`.
- Rebuild the retrieval index after changing knowledge documents if you want retrieval results to reflect those edits.
- The FastAPI app entry point is `src.app.main:app`.
- The root `main.py` is only a lightweight local script entry point.
- `.env` is loaded automatically via `python-dotenv`.

## License

No license file is currently included in this repository.
