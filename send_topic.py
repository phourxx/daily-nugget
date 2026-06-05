#!/usr/bin/env python3
"""
Daily advanced software-engineering topic -> email.

No infra: runs on a GitHub Actions cron, keeps state in topics_log.json
(committed back to the repo) so topics don't repeat and old ones can
resurface as recall questions.

Required env vars (set as GitHub repo secrets):
  ANTHROPIC_API_KEY
  RESEND_API_KEY
  EMAIL_TO        e.g. you@example.com
  EMAIL_FROM      e.g. "Daily SWE <onboarding@resend.dev>"  (resend.dev works with no domain)
"""

import datetime as dt
import json
import os
import sys
import urllib.request

LOG_FILE = "topics_log.json"
DOMAINS_FILE = "domains.txt"   # one domain per line; '#' comments and blanks ignored
# Fallback if domains.txt is missing. List a domain twice to make it twice as frequent.
DEFAULT_DOMAINS = [
    "System Design",
    "Databases",
    "Design Tradeoffs",
    "Distributed Systems",
    "Concurrency & Parallelism",
    "Performance & Scalability",
    "API & Interface Design",
    "Reliability & Failure Modes",
]
PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# OpenAI's lineup changes often — set OPENAI_MODEL to a string you have access to
# (e.g. gpt-5.5, gpt-5.2-chat-latest, gpt-4.1). The default below may be stale.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2-chat-latest")
RECENT_WINDOW = 30          # don't repeat a topic seen in the last N entries
RECALL_AFTER_DAYS = 7       # a topic becomes eligible for recall once it's this old
RECALL_EVERY = 3            # roughly: include a recall question every Nth day


def load_log() -> list[dict]:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_log(log: list[dict]) -> None:
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def load_domains() -> list[str]:
    if os.path.exists(DOMAINS_FILE):
        with open(DOMAINS_FILE) as f:
            domains = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        if domains:
            return domains
    return DEFAULT_DOMAINS


def pick_domain(log: list[dict], domains: list[str]) -> str:
    # Round-robin by entry count: guarantees even coverage, no random clumping.
    return domains[len(log) % len(domains)]


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def anthropic_call(system: str, user: str) -> str:
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"anthropic-version": "2023-06-01", "x-api-key": os.environ["ANTHROPIC_API_KEY"]},
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1200,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    )
    return "".join(b["text"] for b in data["content"] if b["type"] == "text").strip()


def openai_call(system: str, user: str) -> str:
    # max_completion_tokens (not max_tokens) for GPT-5/o-series compatibility;
    # temperature omitted because some reasoning models reject a custom value.
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        {
            "model": OPENAI_MODEL,
            "max_completion_tokens": 1200,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    return data["choices"][0]["message"]["content"].strip()


def llm_call(system: str, user: str) -> str:
    if PROVIDER == "openai":
        return openai_call(system, user)
    if PROVIDER == "anthropic":
        return anthropic_call(system, user)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER!r} (use 'anthropic' or 'openai')")


def pick_recall(log: list[dict]) -> dict | None:
    cutoff = dt.date.today() - dt.timedelta(days=RECALL_AFTER_DAYS)
    eligible = [e for e in log if dt.date.fromisoformat(e["date"]) <= cutoff]
    if not eligible:
        return None
    # deterministic-ish rotation: oldest unreviewed first
    eligible.sort(key=lambda e: (e.get("recalled_count", 0), e["date"]))
    return eligible[0]


def generate_topic(domain: str, recent_titles: list[str]) -> dict:
    avoid = "\n".join(f"- {t}" for t in recent_titles) or "(none yet)"
    system = (
        "You write a single advanced software-engineering micro-lesson for a "
        "senior engineer (~10 yrs, backend-heavy: Python/Django, distributed "
        "systems, fintech). Pick something genuinely non-obvious — a subtle "
        "failure mode, a design tradeoff, an internals detail — not intro "
        "material. Be concrete and opinionated."
    )
    user = (
        f"Today's domain is: {domain}.\n"
        f"Choose a specific, non-obvious topic strictly within that domain.\n\n"
        f"Avoid these recently-covered topics:\n{avoid}\n\n"
        "Return ONLY valid JSON, no markdown fences, with keys:\n"
        '  "title": short topic name\n'
        '  "concept": 2-3 sentences explaining it\n'
        '  "why_it_matters": 1-2 sentences on the practical stakes\n'
        '  "failure_mode": a concrete way engineers get this wrong\n'
        '  "test_question": one question that checks real understanding\n'
    )
    raw = llm_call(system, user).removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def render_email(topic: dict, domain: str, recall: dict | None) -> str:
    parts = [
        f"<p style='margin:0;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#888'>{domain}</p>",
        f"<h2 style='margin:2px 0 4px'>{topic['title']}</h2>",
        f"<p>{topic['concept']}</p>",
        f"<p><b>Why it matters:</b> {topic['why_it_matters']}</p>",
        f"<p><b>Where it goes wrong:</b> {topic['failure_mode']}</p>",
        f"<p><b>Test yourself:</b> {topic['test_question']}</p>",
    ]
    if recall:
        parts.append("<hr style='margin:20px 0;border:none;border-top:1px solid #ddd'>")
        parts.append(
            f"<p style='color:#555'><b>Recall ({recall['date']}):</b> "
            f"{recall['title']} — {recall['test_question']}</p>"
        )
    return "<div style='font-family:system-ui,sans-serif;max-width:600px;line-height:1.5'>" + "".join(parts) + "</div>"


def send_email(html: str, subject: str) -> None:
    body = json.dumps({
        "from": os.environ["EMAIL_FROM"],
        "to": [os.environ["EMAIL_TO"]],
        "subject": subject,
        "html": html,
    }).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Resend returned {resp.status}: {resp.read().decode()}")


def main() -> int:
    log = load_log()
    domains = load_domains()
    domain = pick_domain(log, domains)
    recent_titles = [e["title"] for e in log[-RECENT_WINDOW:]]

    topic = generate_topic(domain, recent_titles)

    recall = pick_recall(log) if len(log) % RECALL_EVERY == 0 else None
    if recall:
        recall["recalled_count"] = recall.get("recalled_count", 0) + 1

    html = render_email(topic, domain, recall)
    subject = f"SWE · {domain} · {topic['title']}"
    send_email(html, subject)

    log.append({
        "date": dt.date.today().isoformat(),
        "domain": domain,
        "title": topic["title"],
        "test_question": topic["test_question"],
        "recalled_count": 0,
    })
    save_log(log)
    print(f"Sent [{domain}]: {topic['title']}" + (f"  (+recall: {recall['title']})" if recall else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
