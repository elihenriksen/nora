# Nora

Nora is a multi-agent harness where each agent represents an area of your life and develops a distinct identity through use. **Personal Council**, the headline feature, convenes your agents to debate a decision together.

**Full design and rationale:** [Introducing Personal Council](https://medium.com/@elivhenriksen/introducing-personal-council-b1a782b92126)

---

## Quickstart

### 1. You need both API keys

Nora uses **two model providers**:

- **Anthropic** (Claude) — for agent voice. Voice quality in sustained, in-character dialogue is the reason.
- **OpenAI** — for the World Engine, moderator, and structured-output checks. Requires schema-validated JSON, which OpenAI's structured-outputs API enforces at the decoding level.

You can technically run with only an OpenAI key by setting `NORA_MODEL_CHARACTER=gpt-4o-mini` in your `.env`, but the agent voice noticeably degrades. **The intended experience requires both keys.**

Get them:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/settings/keys

### 2. Install

**macOS:**

```bash
git clone <your-fork> nora
cd nora
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env       # then edit .env with your keys
./Nora.command
```

First run, macOS may warn it's from an "unidentified developer" — right-click `Nora.command` in Finder and choose **Open** once to allow it.

**Linux:**

```bash
git clone <your-fork> nora
cd nora
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env       # then edit .env with your keys
./Nora.sh
```

**Windows:**

```cmd
git clone <your-fork> nora
cd nora
python -m venv .venv
.venv\Scripts\pip install -e .
copy .env.example .env
notepad .env
.\Nora.bat
```

**Cross-platform fallback** (works everywhere):

```bash
python -m nora             # interactive launcher menu
python -m nora.cli --help  # plain CLI
```

### 3. First run

The launcher creates the world database on first launch and offers to seed four richly-written starter agents — **Charlie** (fishing and hobbies), **Candy** (health and wellbeing), **Gabriel** (home and routines), **Finn** (technology and systems). Say yes the first time so you have something to talk to. They demonstrate what a thoughtful seed enables; one-line interests produce a flatter result.

If something doesn't work, run `nora doctor` to check your environment, keys, and database.

---

## The launcher menu

`./Nora.command` (or `python -m nora`) opens an interactive menu:

| group | option | what it does |
|---|---|---|
| **conversations** | 1 | Chat — pick one agent for 1:1, or several for multi-agent |
| | 2 | Convene a Personal Council |
| | 3 | Watch agents talk autonomously |
| **history** | 4 | History — resume a recent conversation or search across messages |
| **create** | 5 | Create a new agent |
| **inspect** | 6 | Show the world (overview) |
| | 7 | Inspect a pair of agents (their shared topics) |
| | 8 | Show one agent in full (formed views, relationships) |
| | 9 | List postcards |
| **evaluate** | 10 | Run an emergence eval (single probe) |
| | 11 | Run the full eval suite (regression + validation pass) |
| **maintain** | 12 | Consolidate agent memory |
| | 13 | Show LLM usage and cost |
| | 14 | Run a health check (doctor) |
| | 15 | Change your name |
| | 16 | Reset world (delete everything) |

---

## Inside a chat

| command | what it does |
|---|---|
| `/done` | End the conversation and run the World Engine (recommended) |
| `/exit` | End without processing (no updates land) |
| `/skip` | Let the agents take another turn without you saying anything |
| `/info` | Show who's in the room |
| `/file <path>` | Read a text file and send its contents as your message |
| `/kick <name>` | Remove an agent from a multi-agent conversation |
| `@name <text>` | Address a specific agent directly |

`/file` is useful when you want to paste in a long document, code, or a draft you want feedback on without typing it inline.

---

## CLI reference

Every menu action is also a direct command:

```bash
nora init                            # create or upgrade the world DB
nora doctor                          # health check (env, keys, db, fixtures)
nora --version                       # show version

nora agent new                       # interactive agent creation
nora agent list
nora agent show NAME
nora agent delete NAME

nora chat NAME [NAME...]             # 1:1 or multi-agent
nora chat --resume <conv-id>         # resume by id
nora resume                          # pick a recent conversation
nora council "QUESTION" --with N1,N2 # Personal Council
nora autonomous --with all [--topic "..."] [--max-turns N]

nora world                           # overview
nora pair NAME NAME                  # shared context between two agents
nora postcards                       # list postcards
nora search "term"                   # full-text search across messages

nora eval emergence NAME --topic T --prompt "..."   # single probe
nora eval suite [--agent NAME]                      # full suite + validation

nora consolidate [--agent NAME]      # memory consolidation pass
nora usage                           # token + cost telemetry
```

All commands take `--help` for details. Expensive ones (`eval suite`, `consolidate`) print a cost estimate and ask for confirmation; pass `--yes` to skip.

---

## Models & cost

Defaults:

| call | default model | provider |
|---|---|---|
| agent voice | `claude-sonnet-4-6` | Anthropic |
| world engine | `gpt-4o` | OpenAI |
| moderator + wants-to-speak | `gpt-4o-mini` | OpenAI |

Streaming and Anthropic prompt caching are on by default. Turn 2+ in the same session is noticeably faster and ~90% cheaper on the cached portion.

A substantial multi-agent session typically runs under twenty cents. A full eval suite runs about a dime. `nora usage` shows totals broken down by model and purpose.

Pricing in `nora/llm.py` (`PRICING`) is approximate and current as of the model IDs in `.env.example`. Update if rates shift.

---

## Storage

Single SQLite file: `nora.db` at the project root. Delete it (or use menu option 16) to reset the world. Open it with any SQLite viewer to inspect raw state. The schema is in [`nora/schema.sql`](nora/schema.sql) — single source of truth.

WAL mode is enabled for concurrent readers. The chat REPL uses autocommit so long LLM calls don't block telemetry inserts.

---

## Debug mode

Any launcher accepts `--debug` to enable verbose logging for the session (propagates to subprocesses):

```bash
./Nora.command --debug
./Nora.sh --debug
.\Nora.bat --debug
python -m nora --debug
```

---

## Where to start in the code

If you want to understand the architecture by reading the source, the article above lays it out in order. The corresponding files:

- [`nora/voice.py`](nora/voice.py) — system-prompt construction from relational context
- [`nora/world_engine.py`](nora/world_engine.py) — post-conversation processing into structured updates
- [`nora/emergence.py`](nora/emergence.py) — the eval and validation instrument
- [`nora/schema.sql`](nora/schema.sql) — the social ontology, in DDL

---

## License

MIT — see [LICENSE](LICENSE).
