# Vera++ — magicpin AI Challenge submission

## Approach
**Composer (`composer.py`)** turns the four contexts into one WhatsApp message in five stages:
1. **Grounded fact extraction.** Only real data reaches the LLM:
   - the digest item the trigger points to (`top_item_id`)
   - the merchant's own numbers, compared with peer stats that are **computed in code** so the LLM never does arithmetic
   - active offers vs. expired ones, signals, review themes, and the last few conversation turns
   - customer relationship and preferences, when a customer is involved
2. **Per-trigger playbook.** All 24 trigger kinds in the dataset map to an *angle*, a *compulsion lever* and a *CTA shape*. For example, `perf_dip` uses loss aversion plus effort externalization with a YES/STOP CTA, and `curious_ask_due` asks the merchant an open question.
3. **LLM.** Groq `gpt-oss-120b` in JSON mode, `temperature=0`, fixed seed. When a model hits its rate limit, the bot moves on to `gpt-oss-20b` and then `qwen3.8-27b`, since free-tier limits are per model. The system prompt contains the rubric, the hard rules (no fabrication, no preamble, one CTA as the last line) and the language rule (Hinglish in Roman script when the merchant or customer prefers `hi`).
4. **Validator.** Checks the draft for taboo vocabulary from the category, generic phrases, length, multiple CTAs, and verbatim repeats. On failure the LLM gets one retry with the problems listed.
5. **Deterministic fallback.** If the LLM is down, slow or has no key, a template still produces a message grounded in the context. The bot therefore always answers within the 30s limit.

**Replies (`conversation_handlers.py`)** are routed by rules first, then the LLM:
- **Auto-reply** (canned-phrase regex, or the same text 3+ times): one nudge aimed at the owner, then end. Counted per merchant, because the judge may open a new conversation for each turn.
- **Hostility:** one de-escalation that restates what Vera does. Repeated abuse ends the conversation.
- **Opt-out:** end immediately.
- **"Later" / "busy":** wait.
- **Intent** ("yes", "go ahead", "haan karo", "I want to join") switches the conversation to **action mode**. In action mode the LLM is told never to ask qualifying questions.
- **Off-topic** (GST, loans, etc.): one polite decline, then steer back to the open thread.
- **Language** is detected on every turn (Devanagari or Roman-Hindi markers), so the bot follows a merchant who switches language.
- **Turn budget:** the conversation closes after 5 bot turns.

**Server (`bot.py`)** is FastAPI and implements all 5 endpoints plus `/v1/teardown`:
- `/v1/context` is idempotent per version; an older version gets a 409.
- `/tick` trusts the judge's `available_triggers` and skips repeated suppression keys. If a customer-scope trigger arrives without the customer's context, Vera asks the merchant to approve the outreach instead of messaging the customer blind. It handles higher urgency first, sends one action per (merchant, conversation), and composes in parallel within a 24s budget (`TICK_BUDGET`).

## Run
```bash
pip install -r requirements.txt
# put GROQ_API_KEY / LLM_MODEL in .env (auto-loaded)
uvicorn bot:app --host 0.0.0.0 --port 8080      # server
python generate_submission.py                     # writes submission.jsonl (30 test pairs)
```
Deployed on Render: **<RENDER_URL>** (`render.yaml` + `Dockerfile`, single instance because state is in memory).

Contract check (all 5 endpoints, the website's curl examples, idempotency/409/400, tick/reply shapes, <30s):
```bash
BOT_URL=http://localhost:8080 python tests/contract_check.py
```
The bot accepts JSON bodies with or without `Content-Type: application/json`, since the website's curl examples omit the header. Malformed bodies get a 400; a malformed `/tick` gets `{"actions": []}`.

## Tradeoffs
- **Rules before the LLM for replies.** Safety-critical routing (auto-reply, opt-out, intent) is deterministic and instant. The LLM only writes content.
- **One LLM call, not an agent loop.** This keeps latency around 1–3s on Groq, well inside the 30s limit.
- **In-memory state.** This is allowed by the brief. It would need Redis for production.

## What would help most
- Real merchant reply transcripts labelled with outcomes, to tune which lever works for each category.
- Open booking slots for more merchants, since customer-facing messages can only offer real slots.
- The WhatsApp template registry, so first messages can map exactly to approved templates.
