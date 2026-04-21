# Support Agent Starter

[Python](https://www.python.org/)
[FastAPI](https://fastapi.tiangolo.com/)
[OpenAI](https://platform.openai.com/docs)
[Tests](https://docs.pytest.org/)
[License](#license)

A small FastAPI customer-support assistant starter that can be adapted for many kinds of businesses. The app loads business-managed JSON documents, builds a support prompt from those documents and conversation history, calls OpenAI, and stores chat transcripts on disk.

## Features

- FastAPI app with health, chat, and streaming chat endpoints.
- OpenAI Responses API integration through an isolated LLM client.
- JSON-backed business profile and support knowledge files that are easy to customize.
- File-backed conversation history by `session_id`.
- Layered structure for API routes, domain logic, infrastructure, and tests.
- Designed to be forked and retuned for a new business with minimal code changes.

## Project Structure

```text
.
├── data/
│   ├── business_profile.json       # Business identity, contact details, tone, metadata
│   ├── knowledge.json              # Support policies, FAQs, and business knowledge
│   └── conversations/              # Stored chat transcripts by session id
├── src/app/
│   ├── api/                        # FastAPI routes and request/response schemas
│   ├── core/                       # Settings and dependency wiring
│   ├── domain/support/             # Prompt building and support service logic
│   └── infrastructure/             # OpenAI, content loading, and storage adapters
├── tests/                          # Unit and route tests
├── main.py                         # Simple local Python entry point
├── pyproject.toml                  # Project metadata and dependencies
└── uv.lock                         # Locked dependency versions
```

## Requirements

- Python 3.12 or newer
- `uv`
- An OpenAI API key for the `/chat` endpoints

## Who This Is For

This repo is intended as a reusable starting point for anyone who wants a support agent tailored to their own business. Fork it, update the JSON content in `data/`, adjust the tone and policies, and you have a basic support API without needing to redesign the whole app.

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
```

Only `OPENAI_API_KEY` is required for real chat responses. The other values have defaults and can be omitted.

## Customizing For Your Business

Most customization happens in the `data/` directory:

- Update `data/business_profile.json` with your business name, support channels, operating hours, escalation rules, and tone.
- Update `data/knowledge.json` with your policies, FAQs, services, product details, and handoff rules.
- Keep the app code as-is if your needs fit the current API shape.
- Extend the domain or API layers later if you want business-specific workflows.

The goal is to let each fork keep the same support-agent foundation while swapping in business-specific content and behavior.

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

Send a chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your support hours?", "session_id": "demo-session"}'
```

Stream a chat response:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you help me with a refund?", "session_id": "demo-session"}'
```

## Business Documents

The assistant reads business content from the `data/` directory. These files are intended to be edited in each fork so the agent reflects that business's identity, policies, and support knowledge.

### Business Profile

Edit `data/business_profile.json` for business identity and support operations:

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

### Knowledge Base

Edit `data/knowledge.json` for policies, FAQs, product notes, service details, handoff rules, and other support knowledge:

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

Each key under `sections` becomes a named knowledge section in the system prompt. Each value must be a JSON array of strings.

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

### Tenant-Specific Documents

The loaders support tenant-specific files even though the current chat API does not expose `tenant_id` yet. If domain code passes a tenant id, files are resolved like this:

```text
data/{tenant_id}/business_profile.json
data/{tenant_id}/knowledge.json
```

If a tenant-specific file is missing, the app falls back to:

```text
data/business_profile.json
data/knowledge.json
```

## Conversation Storage

Conversations are stored as JSON files in:

```text
data/conversations/
```

The filename is based on `session_id`. For example, `session_id: "demo-session"` is stored at:

```text
data/conversations/demo-session.json
```

If no `session_id` is provided, the app generates one internally. The current HTTP response only returns the assistant response, so provide your own `session_id` when you want to continue a conversation across requests.

## Configuration

Settings are loaded from environment variables and `.env`.


| Variable                   | Default              | Purpose                                 |
| -------------------------- | -------------------- | --------------------------------------- |
| `APP_NAME`                 | `Support API`        | FastAPI application title               |
| `ENVIRONMENT`              | `development`        | Runtime environment label               |
| `API_V1_PREFIX`            | empty                | Reserved API prefix setting             |
| `CONTENT_DATA_DIR`         | `data`               | Directory containing business documents |
| `CONVERSATION_STORAGE_DIR` | `data/conversations` | Directory for saved transcripts         |
| `OPENAI_API_KEY`           | unset                | Required for chat endpoints             |
| `OPENAI_MODEL`             | `gpt-4.1-mini`       | OpenAI model used by the LLM client     |


## API Endpoints


| Method | Path           | Description                            |
| ------ | -------------- | -------------------------------------- |
| `GET`  | `/`            | Basic app verification                 |
| `GET`  | `/health`      | Health check                           |
| `POST` | `/chat`        | Non-streaming support chat             |
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

Run the test suite:

```bash
uv run pytest
```

The tests cover content loaders, prompt building, support policies, conversation storage, OpenAI client translation, service behavior, and API routes.

## Development Notes

- Keep runtime business content in `data/business_profile.json` and `data/knowledge.json`.
- Treat the JSON files in `data/` as the main customization layer for each fork.
- Keep local secrets in `.env`; `.env` is ignored by Git.
- Keep generated or test conversation transcripts in `data/conversations/`.
- The app entry point for the API is `src.app.main:app`.
- The root `main.py` is only a simple script entry point and is not the FastAPI server.

## License

No license file is currently included in this repository.