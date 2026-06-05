# scripts/

Optional scripts. **All of these except `seed_characters.py` make real LLM
API calls and cost money.**

## Reproducible setup

- `seed_characters.py` — create the four starter agents (Charlie, Candy,
  Gabriel, Finn) with rich, vetted seeds. No LLM calls.
  ```
  .venv/bin/python scripts/seed_characters.py
  .venv/bin/python scripts/seed_characters.py --reset   # delete db first
  ```

## Demos (these spend money)

- `demo_basic_chat.py` — runs an end-to-end multi-agent conversation,
  then the World Engine, then prints what was learned.
- `demo_council_and_emergence.py` — runs a Personal Council on a question,
  then the emergence eval. Assumes Charlie + Candy already exist.
- `demo_three_speakers.py` — verifies the moderator is picking all three
  agents on substantive prompts.
- `demo_voice_register.py` — sends a casual prompt ("do I get chipotle")
  to confirm agents match register and don't moralize.
- `demo_voice_register_substantive.py` — same as above but with a real
  decision prompt.
- `demo_strip_metaphor.py` — A/B test: same agent with metaphor in the
  seed vs without, to demonstrate that voice quirks come from the seed,
  not the engine.

Run one with `.venv/bin/python scripts/<name>.py`. Costs are typically
under $0.10 each but compound if you run them all.
