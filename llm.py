"""Provider chain and roles for article-writer, on book writer's shared AI suite.

Two roles are used across the pipeline:
  PRO   - topic selection, outline audit, review, final approval (better judgement)
  FLASH - research synthesis, outlining, drafting, rewriting (high volume)
A third pool (JUDGES) scores whether a text reads as AI-written: a fixed panel
(``JUDGE_MODELS``), falling back to the drafting model when every judge fails.

``AW_BACKEND`` is an ordered chain, not a single name: ``claude,hyper,go``
means "Claude Code CLI, and if it fails, hyper, and if that fails too, go".
Each provider carries its own catalogue, so the roles are translated as the call
walks down the chain — PRO on ``claude`` is ``opus``, PRO on ``hyper`` is
``qwen3.8-max``. Pick it interactively in the wizard, with ``--provider`` /
``--pro`` / ``--flash`` / ``--backups``, or in ``.env``.

The judges never follow the chain into ``claude``: an exclusive provider only
ever serves models from its own catalogue.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import ui

# On Windows, subprocess.run() and blocking network I/O swallow Ctrl+C because
# Python defers signal delivery until the C call returns. This handler raises
# immediately so the user never waits for a 660s LLM call to finish before the
# interrupt lands.
if sys.platform == "win32":
    def _sigint_handler(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

BASE_URL = os.environ.get("AW_BASE_URL", "https://hyper.charm.land/v1")

# The `go` link is opencode-go. Version suffixes are a hyper-ism
# (deepseek-v4-pro-0813); opencode-go carries the unsuffixed name.
OPENCODE_PROVIDER = "opencode-go"

# Claude Code takes the short aliases; anything else is passed through, so a full
# id (claude-opus-5) also works.
CLAUDE_ALIAS = {"pro": "opus", "flash": "sonnet",
                "qwen3.8-max": "opus", "deepseek-v4-pro-0813": "sonnet"}

OPENCODE_ALIAS = {
    "deepseek-v4-pro-0813": "deepseek-v4-pro",
    "deepseek-v4-flash-0731": "deepseek-v4-flash",
}

ROOT = pathlib.Path(__file__).resolve().parent

# openai-oauth runs a local proxy that handles ChatGPT OAuth login and exposes
# an OpenAI-compatible /v1 surface. The proxy is started on demand via npx.
_OAUTH_PORT = 10531
_OAUTH_BASE_URL = f"http://127.0.0.1:{_OAUTH_PORT}/v1"

# Estimated API-equivalent costs (USD per million tokens: input/output) and the
# minimum ChatGPT tier that serves each model. Used to show value in the wizard.
_OAUTH_COSTS: dict[str, tuple[float, float, str]] = {
    "gpt-6-astra":    (10.0, 50.0, "Pro $100"),
    "gpt-6-sol":      (2.0, 10.0, "Plus $20"),
    "gpt-6-luna":     (0.10, 0.50, "Free"),
    "gpt-5.6-sol":    (4.0, 20.0, "Plus $20"),
    "gpt-5.6-terra":  (2.0, 10.0, "Plus $20"),
    "gpt-5.6-luna":   (0.5, 2.0,  "Free"),
    "gpt-5.5-instant": (1.0, 4.0, "Plus $20"),
    "gpt-5.5-thinking": (3.0, 12.0, "Plus $20"),
    "gpt-5.4":        (2.5, 10.0, "Pro $100"),
    "gpt-5.4-mini":   (0.15, 0.6, "Plus $20"),
    "o3":             (10.0, 40.0, "Pro $100"),
    "o3-mini":        (1.1, 4.4,  "Plus $20"),
    "o4-mini":        (1.1, 4.4,  "Plus $20"),
}
# Subscription tiers: (name, USD/month, usage multiplier vs Plus)
_OAUTH_TIERS = [
    ("Free", 0, 1),
    ("Go", 8, 1),
    ("Plus", 20, 1),
    ("Pro $100", 100, 5),
    ("Pro $200", 200, 20),
]


def _oauth_proxy_running() -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", _OAUTH_PORT), timeout=1):
            return True
    except OSError:
        return False


def oauth_cost_label(model: str) -> str:
    """Estimated API-equivalent cost and subscription tier for an oauth model."""
    entry = _OAUTH_COSTS.get(model.lower())
    if not entry:
        return ""
    inp, out, tier = entry
    return f"~${inp:.2f}/${out:.2f} per Mtok (API) · requires {tier}"


def oauth_value_summary() -> str:
    """One-line value estimate: what the subscription buys vs. paying per token."""
    # A typical article run costs ~100k tokens; at gpt-6-sol rates that is ~$0.60.
    # 25 articles/month ≈ $15 at API rates vs $20 for Plus.
    return ("Plus $20/mo ≈ $15/mo of tokens at API prices for ~25 articles; "
            "worth it from ~35 articles/month. Pro $100 gives 5× the limit and Astra")

_HYPER_MODELS = ["qwen3.8-max", "qwen3.7-max", "deepseek-v4-pro-0813",
                 "deepseek-v4-flash-0731", "deepseek-v4.1-flash", "qwen3.8-flash",
                 "glm-5.2", "glm-5.3-flash",
                 "kimi-k3", "minimax-m3"]

_ZEN_BASE_URL = os.environ.get("AW_ZEN_URL", "https://opencode.ai/zen/v1")
# Zen routes each family server-side; the OpenAI-compatible /v1 surface carries
# these ids (the GPT/Gemini/Claude ones speak /responses and /messages instead).
_ZEN_MODELS = ["qwen3.8-max", "qwen3.7-max", "qwen3.7-plus",
               "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4.1-flash", "qwen3.8-flash",
               "glm-5.2", "glm-5.1", "glm-5.3-flash", "kimi-k3", "kimi-k2.6", "minimax-m3",
               "big-pickle", "x-preview-f-free"]
ZEN_ALIAS = {"deepseek-v4-pro-0813": "deepseek-v4-pro",
              "deepseek-v4-flash-0731": "deepseek-v4-flash"}

_GROK_BASE_URL = os.environ.get("AW_GROK_URL", "https://api.x.ai/v1")
_GROK_MODELS = ["grok-4", "grok-4-fast", "grok-3", "grok-3-mini"]

# gpt4free's local server (`pip install -U "g4f[api]"`, then `g4f api`): free,
# keyless, g4f's own model names. ai-suite's gpt4free provider serves it.
_G4F_BASE_URL = os.environ.get("AW_G4F_URL", "http://127.0.0.1:1337/v1")
_G4F_MODELS = ["deepseek-v4-pro", "deepseek-v4.1-flash", "glm-5.3", "glm-5.3-flash",
               "kimi-k3", "minimax-m3", "qwen-3.8-2.4t-a95b", "gemini-3.8-flash"]

# exclusive: the provider only serves models from its own catalogue, so a call
# for a model it does not know skips it and walks on down the chain. That is what
# keeps the judges off Claude when Claude is doing the writing.
PROVIDERS: dict[str, dict] = {
    "claude": {"label": "Claude Code CLI — runs on your subscription, no key",
               "pro": "claude-opus-5-5", "flash": "sonnet", "exclusive": True,
               "alias": CLAUDE_ALIAS,
               "models": ["opus", "sonnet", "haiku",
                          "claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5",
                          "claude-haiku-4-5-20251001"]},
    "hyper":  {"label": "hyper.charm.land — API OpenAI-compatible (AW_API_KEY)",
               "pro": "qwen3.8-flash", "flash": "deepseek-v4.1-flash",
               "exclusive": False, "alias": {}, "models": _HYPER_MODELS,
               "base_url": BASE_URL, "key_env": "AW_API_KEY"},
    "zen":    {"label": "OpenCode Zen — direct API at opencode.ai/zen (OPENCODE_API_KEY)",
               "pro": "glm-5.3-flash", "flash": "deepseek-v4.1-flash",
               # gpt/gemini/claude ids on zen speak /responses and /messages,
               # not the chat-completions surface this module uses.
               "skip_live": r"^(gpt|gemini|claude|o\d)",
               "exclusive": False, "alias": ZEN_ALIAS, "models": _ZEN_MODELS,
               "base_url": _ZEN_BASE_URL, "key_env": "OPENCODE_API_KEY"},
    "grok":   {"label": "xAI Grok — direct API at x.ai (XAI_API_KEY)",
               "pro": "grok-4", "flash": "grok-4-fast",
               "exclusive": False, "alias": {}, "models": _GROK_MODELS,
               "base_url": _GROK_BASE_URL, "key_env": "XAI_API_KEY"},
    "g4f":    {"label": "gpt4free — local `g4f api` server, free, no key",
               "pro": "deepseek-v4-pro", "flash": "glm-5.3",
               "exclusive": False, "alias": {}, "models": _G4F_MODELS,
               # g4f hands any bearer but its own G4F_API_KEY to its backends.
               "base_url": _G4F_BASE_URL, "key_env": "G4F_API_KEY",
               "own_key_only": True, "keyless": True},
    "go": {"label": "opencode CLI local — contra opencode-go",
           "pro": "qwen3.8-flash", "flash": "deepseek-v4.1-flash",
           "exclusive": False, "alias": OPENCODE_ALIAS, "models": _HYPER_MODELS},
    "oauth":  {"label": "openai-oauth — local proxy on your ChatGPT account",
               "pro": "gpt-6-sol", "flash": "gpt-6-luna",
               "exclusive": True, "alias": {},
               "models": ["gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4", "gpt-5.4-mini",
                          "o3", "o3-mini", "o4-mini"],
               "base_url": _OAUTH_BASE_URL},
}


_catalogue_cache: dict[str, list[str]] = {}


# The CLI route used to be called `opencode`; .env files still say so, and
# silently dropping the name there would lose the chain's last link.
def _canon(backend: str) -> str:
    return "go" if backend == "opencode" else backend


def set_models(backend: str, pro: str = "", flash: str = "") -> None:
    """Override a provider's own PRO/FLASH pair — the models it answers with when
    it is a *backup*. `_as()` reads the pair straight off the spec, so writing it
    here is the whole mechanism."""
    spec = PROVIDERS[backend]
    if pro:
        spec["pro"] = pro
    if flash:
        spec["flash"] = flash


def parse_models(raw: str) -> dict[str, tuple[str, str]]:
    """`hyper:qwen3.8-max/deepseek-v4-pro-0813,zen:glm-5.2/deepseek-v4-pro`.

    FLASH may be omitted (`zen:glm-5.2`); an unknown provider is ignored rather
    than raising, because this also parses a hand-edited .env line.
    """
    out: dict[str, tuple[str, str]] = {}
    for chunk in raw.split(","):
        backend, _, pair = chunk.strip().partition(":")
        backend = _canon(backend.strip().lower())
        if backend not in PROVIDERS or not pair.strip():
            continue
        pro, _, flash = pair.partition("/")
        out[backend] = (pro.strip(), flash.strip())
    return out


for _b, (_pro, _flash) in parse_models(os.environ.get("AW_MODELS", "")).items():
    set_models(_b, _pro, _flash)


def _default_chain() -> list[str]:
    raw = os.environ.get("AW_BACKEND", "hyper")
    chain = [_canon(b.strip().lower()) for b in raw.split(",")]
    chain = [b for b in chain if b in PROVIDERS]
    # go has always been the implicit last resort; AW_OPENCODE_FALLBACK=0
    # is the old switch that turns it off, and it still does.
    if (len(chain) == 1 and "go" not in chain
            and os.environ.get("AW_OPENCODE_FALLBACK", "1") != "0"):
        chain.append("go")
    return chain or ["hyper", "go"]


CHAIN = _default_chain()
# The provider PRO/FLASH were picked from. Callers may rotate CHAIN (a drafted
# section handed to a backup), and the new head still needs the role translated.
HEAD = CHAIN[0]
PRO = os.environ.get("AW_MODEL_PRO") or PROVIDERS[CHAIN[0]]["pro"]
FLASH = os.environ.get("AW_MODEL_FLASH") or PROVIDERS[CHAIN[0]]["flash"]


# Fixed detector panel since 2026-09-26: opus 5.5 took kimi-k3's seat and gpt-6-astra
# joined. Each lives on an exclusive provider (claude, oauth), so `_serves` sends it
# straight there. A judge that fails is skipped; if all fail, `humanize.llm_judges`
# falls back to the drafting model.
JUDGE_MODELS = ["claude-opus-5-5", "gpt-6-astra"]


def _judges() -> list[str]:
    return list(JUDGE_MODELS)


JUDGES = _judges()


def configure(chain: list[str] | None = None, pro: str = "", flash: str = "",
              models: dict[str, tuple[str, str]] | None = None) -> None:
    """Repoint the router at runtime — what the wizard and the CLI flags call.

    Models default to the head provider's own pair, so switching provider without
    naming models does the sensible thing instead of sending `opus` to hyper.
    """
    global CHAIN, HEAD, PRO, FLASH, JUDGES
    for backend, (p, f) in (models or {}).items():
        set_models(backend, p, f)
    if chain:
        picked = [_canon(b) for b in chain if _canon(b) in PROVIDERS]
        if not picked:
            raise ValueError(f"proveedor desconocido: {chain}")
        CHAIN = list(dict.fromkeys(picked))
    HEAD = CHAIN[0]
    PRO = pro or PROVIDERS[CHAIN[0]]["pro"]
    FLASH = flash or PROVIDERS[CHAIN[0]]["flash"]
    JUDGES = _judges()


def _serves(backend: str, model: str) -> bool:
    prov = PROVIDERS[backend]
    if model in (PRO, FLASH):
        return True
    if prov["exclusive"]:
        return model in prov["models"]
    # A model an exclusive provider owns (the judges) is not tried on open gateways.
    return not any(p["exclusive"] and model in p["models"] for p in PROVIDERS.values())


def _as(backend: str, model: str) -> str:
    """The name this provider knows the model by."""
    prov = PROVIDERS[backend]
    # Role translation is for backups only: the head provider got PRO/FLASH by
    # their literal ids straight from its own catalogue, so a picked model
    # (x-preview-f-free) must not be swapped back to the default (qwen3.7-max).
    if backend != HEAD:
        if model == PRO:
            model = prov["pro"]
        elif model == FLASH:
            model = prov["flash"]
    # Alias last, not instead: opencode's catalogue is spelled the hyper way, so
    # the role lands on `deepseek-v4-pro-0813` and still needs unsuffixing.
    return prov["alias"].get(model, model)


def describe() -> str:
    backups = " ".join(f"{b}={PROVIDERS[b]['pro']}/{PROVIDERS[b]['flash']}"
                       for b in CHAIN[1:])
    judges = "judge" if len(JUDGES) == 1 else "judges"
    return (f"{' → '.join(CHAIN)} | PRO={PRO} FLASH={FLASH}"
            + (f" | backups: {backups}" if backups else "")
            + f" | {judges}={', '.join(JUDGES)}")


def _live_models(backend: str) -> list[str]:
    """Model ids the provider serves right now, [] if it cannot be asked."""
    spec = PROVIDERS[backend]
    try:
        if backend == "go":
            exe = shutil.which("opencode")
            if not exe:
                return []
            proc = subprocess.run([exe, "models"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30)
            # The suffixed hyper-spelling is a go-ism; unsuffix back to the
            # catalogue's names.
            rev = {v: k for k, v in OPENCODE_ALIAS.items()}
            prefix = OPENCODE_PROVIDER + "/"
            return [rev.get(t[len(prefix):], t[len(prefix):])
                    for line in proc.stdout.splitlines() for t in line.split()
                    if t.startswith(prefix)]
        if backend == "oauth":
            if not _oauth_proxy_running():
                return []
        base = spec.get("base_url")
        if not base:
            return []
        import requests
        key = _key_for(spec)
        r = requests.get(base.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {key}"} if key else {},
                         timeout=15)
        # g4f also lists each of its backends (provider: true) and image models.
        ids = [str(m["id"]) for m in r.json().get("data", [])
               if m.get("id") and not m.get("provider") and not m.get("image")]
        skip = spec.get("skip_live")   # families this endpoint serves another way
        return [i for i in ids if not (skip and re.search(skip, i))]
    except Exception:  # noqa: BLE001 - a dead catalogue degrades to the built-in list
        return []


def catalogue(backend: str) -> list[str]:
    """Every model the provider offers: the built-in list plus whatever its
    /models endpoint (or CLI) answers right now.

    The built-in lists go stale the week a new model ships, and a model name a
    provider does not know is a 404 — so the wizard offers the live list merged
    over the built-in one, deduped, built-ins first.

    When the live list answers it is also the truth about what still exists:
    built-ins the provider dropped are pruned instead of sitting at the top of
    the wizard menu forever (zen removed qwen3.7-max server-side while its own
    default still pointed at it — every default pick 401'd). A dead catalogue
    degrades to the untouched built-in list.
    """
    if backend not in _catalogue_cache:
        built = list(PROVIDERS[backend]["models"])
        live = _live_models(backend)
        if live:
            living = set(live)
            built = [m for m in built if m in living]
            live = sorted(living.difference(built))
        _catalogue_cache[backend] = built + live
    return _catalogue_cache[backend]


def _opencode_auth_key() -> str:
    """The key opencode's CLI already stores at login — the same one zen needs.
    Crush and every opencode tool read it here, so the script may too instead of
    demanding a duplicate OPENCODE_API_KEY in .env."""
    p = pathlib.Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        auth = json.loads(p.read_text(encoding="utf-8"))
        for name in ("opencode-zen", "opencode", "opencode-go"):
            entry = auth.get(name) or {}
            if entry.get("type") == "api" and entry.get("key"):
                return str(entry["key"])
    except Exception:  # noqa: BLE001 - no auth file is just no fallback
        pass
    return ""


def _key_for(spec: dict) -> str:
    # A custom endpoint must never inherit a different provider's credential.
    if spec.get("own_key_only"):
        return os.environ.get(spec.get("key_env", ""), "")
    # The oauth proxy handles real auth; the OpenAI client just needs a non-empty
    # bearer so it does not refuse to initialise.
    if spec.get("base_url") == _OAUTH_BASE_URL:
        return "openai-oauth"
    # Zen before the shared fallbacks: AW_API_KEY is hyper's key and a wrong
    # bearer here looks like a dead provider instead of a misrouted credential.
    if spec.get("key_env") == "OPENCODE_API_KEY":
        key = os.environ.get("OPENCODE_API_KEY") or _opencode_auth_key()
        if key:
            return key
    return (os.environ.get(spec.get("key_env", ""))
            or os.environ.get("AW_API_KEY")
            or os.environ.get("OPENAI_API_KEY"))


# Claude Code prints a quota notice on **stdout and exits 0**, so it arrives
# looking exactly like a completion: «You've hit your session limit · resets
# 9:10am». It got drafted into the article and parsed as JSON. The length guard
# is what keeps an article that discusses limits from being thrown away — a
# quota notice is one line, never a section.
_QUOTA = re.compile(r"(?i)(session|usage|rate) limit|limit reached|"
                    r"resets? (at )?\d|upgrade to (pro|max)|failed to authenticate|"
                    r"authentication failed|oauth session expired|not logged in")


def _quota_notice(out: str) -> bool:
    return len(out) < 400 and bool(_QUOTA.search(out))


# Every completion runs on book writer's AIService -- the AI suite the whole
# workspace shares (music writer, mathforge, book-watch, lamplight...). This
# module keeps what is article-writer's own: the provider chain, PRO/FLASH roles
# and their per-provider translation, catalogues, key discovery, heartbeat, the
# interactive recovery and JSON repair. AW_BOOK_WRITER overrides the path.
BOOK_WRITER = pathlib.Path(os.environ.get("AW_BOOK_WRITER")
                           or ROOT.parent / "book writer")

# article-writer's backend names -> book writer's providers. Custom endpoints from the
# game's settings ride the generic OpenAI-compatible client ("openrouter").
SHARED = {"claude": "claude", "hyper": "hyper", "zen": "opencode-zen",
          "grok": "grok", "go": "opencode-go", "oauth": "openai-oauth", "g4f": "gpt4free"}

_services: dict[tuple, Any] = {}
_services_lock = threading.Lock()


def shared_service(provider: str, overrides: dict[str, Any]) -> Any:
    """book writer's AIService for `provider`, with article-writer's key/endpoint layered
    on top. One instance per distinct setting, shared by every thread."""
    cache_key = (provider, tuple(sorted(overrides.items())))
    with _services_lock:
        if cache_key not in _services:
            if str(BOOK_WRITER) not in sys.path:
                sys.path.insert(0, str(BOOK_WRITER))
            from ai_book_creator.cli import provider_config_path
            from ai_book_creator.services.ai_service import AIService
            _services[cache_key] = AIService(
                config_path=provider_config_path(provider),
                usage_state_path=str(ROOT / "output" / "shared_ai_usage.json"),
                allow_auth_prompt=False, client_max_retries=0,
                config_overrides=dict(overrides))
        return _services[cache_key]


def _send(backend: str, model: str, prompt: str, system: str | None, *,
          temperature: float | None, max_tokens: int | None, retries: int,
          wait: bool = False) -> str:
    """One link of the chain: `model` (already in this provider's spelling) on `backend`."""
    spec = PROVIDERS[backend]
    provider = SHARED.get(backend, "openrouter")
    if provider == "claude":
        # The Claude Code CLI on the user's subscription: no key, no endpoint.
        overrides: dict[str, Any] = {"timeout": 1800}
    else:
        # Streamed, so the timeout is per chunk: a slow reasoning model outlives a
        # gateway's idle timeout on a plain request (zen answered 503 at 207s on a
        # completion the stream finished in 341s, measured 2026-08-23).
        overrides = {"timeout": int(float(os.environ.get("AW_TIMEOUT", "300"))), "stream": True,
                     # No cap_is_ceiling: AIService asks for the model's whole output
                     # allowance, since a cap only truncates and a cut reply costs a retry.
                     "token_param": "max_tokens"}
        if spec.get("base_url") and backend != "go":
            overrides["base_url"] = spec["base_url"]
        if backend not in SHARED:
            overrides["headers"] = {}  # a custom endpoint gets no book-writer attribution headers
        if backend not in ("go", "oauth"):
            # go reads opencode's own login and oauth the local proxy, inside book writer.
            key = _key_for(spec)
            # A keyless local server (g4f) with no key of its own gets no bearer at all.
            if not key and not spec.get("keyless"):
                # RuntimeError, not SystemExit: this provider may be one link in a
                # chain, and a missing key here has to let the next one try.
                raise RuntimeError(f"missing key for {backend}: set "
                                   f"{spec.get('key_env', 'AW_API_KEY')}=sk-... in article-writer/.env")
            if key:
                overrides["api_key"] = key
    out = shared_service(provider, overrides).generate_content(
        prompt, model=model, system=system, temperature=temperature,
        max_completion_tokens=max_tokens, max_retries=retries, wait_for_limits=wait)
    out = (out or "").strip()
    if not out:
        raise RuntimeError(f"{backend}/{model} returned no text")
    if _quota_notice(out):
        raise RuntimeError(f"{backend}/{model} out of quota: {out}")
    return out


# Set by the wizard: an attended run gets to answer for a dead provider instead
# of watching the whole chain fall over. Worker threads never prompt — input()
# from one of those would interleave with the pipeline's own output.
INTERACTIVE = False


def resolve_choice(raw: str, options: list[str]) -> str:
    """Accept an option either by name or by its 1-based index. '' = no match.

    Lives here rather than in ``main`` because ``_recover`` needs it too, and the
    wizard and the recovery prompt must read a menu answer the same way.
    """
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw if raw in options else ""


def _recover() -> None:
    """Interactive recovery after every link of the chain failed: hold or change."""
    while True:
        raw = input("\n[w]ait 30s and retry, [c]hange provider/model, "
                    "[a]bort? ").strip().lower()
        if raw.startswith("w"):
            time.sleep(30)
            return
        if raw.startswith("a"):
            raise KeyboardInterrupt("aborted by the user")
        if raw.startswith("c"):
            provs = list(PROVIDERS)
            print("\nProviders:")
            for i, k in enumerate(provs, 1):
                print(f"  {i}) {k:<9} {PROVIDERS[k]['label']}")
            prov = resolve_choice(input(f"Provider [{'|'.join(provs)}]: ").strip(), provs)
            while not prov:
                print(f"Invalid. Pick 1-{len(provs)} or one of {provs}.")
                prov = resolve_choice(input(f"Provider [{'|'.join(provs)}]: ").strip(), provs)
            # Lazy import: main imports llm, so a module-level import would be
            # circular. pick_pair prints the provider's live catalogue and asks
            # for PRO/FLASH exactly like the wizard does.
            pro, flash = __import__("main").pick_pair(prov)
            rest = [b for b in CHAIN[1:] if b != prov]
            configure([prov, *rest], pro, flash)
            print(f"\n→ {describe()}\n")
            return


# Seconds between «still running» ticks while a model call holds the console.
# A single completion can take minutes and prints nothing, which is
# indistinguishable from a hang; this says who owns the silence and for how long.
HEARTBEAT = int(os.environ.get("AW_HEARTBEAT", "60"))

_beats: list[threading.Event] = []


def _stop_all_beats() -> None:
    # A daemon tick printing into a dying interpreter aborts shutdown with
    # «could not acquire lock for <stdout>»; every ticker is silenced at exit.
    for ev in _beats:
        ev.set()


atexit.register(_stop_all_beats)


def _heartbeat(label: str) -> threading.Event:
    stop = threading.Event()
    _beats.append(stop)

    def tick():
        # Ticks only while the call is genuinely still running: once the event
        # is set the loop must exit without printing, or completion spins the
        # counter in a tight loop. No tick limit: it used to stop at ten, and
        # opencode (900s per attempt, 1800s with the sibling retry) and claude
        # (1800s) both outlive that — the call went silent right when the wait
        # started looking like a hang.
        waited = 0
        while not stop.wait(HEARTBEAT):
            waited += HEARTBEAT
            try:
                ui.status(f"[llm] {label}: still running ({waited}s)…")
            except Exception:  # noqa: BLE001 - a dying console must not raise
                return

    threading.Thread(target=tick, daemon=True).start()
    return stop


def chat(model: str, prompt: str, system: str | None = None, *,
         temperature: float = 0.8, max_tokens: int | None = None,
         retries: int = 4, fallback: bool = True) -> str:
    """One-shot completion, walking the provider chain until one answers."""
    errors: list[str] = []
    for i, backend in enumerate(CHAIN):
        if i and not fallback:
            break
        if not _serves(backend, model):
            continue
        target = _as(backend, model)
        try:
            ui.status(f"[llm] {model} → {backend}/{target} "
                   f"({len(prompt.split()):,}-word prompt)…")
            started = time.time()
            beat = _heartbeat(target)
            try:
                # A usage limit walks on to the next link; only the last link, with
                # nobody at the console to repoint it, waits the limit out.
                last = i + 1 == len(CHAIN) or not fallback
                out = _send(backend, target, prompt, system, temperature=temperature,
                            max_tokens=max_tokens, retries=retries,
                            wait=last and not INTERACTIVE)
            finally:
                beat.set()
            ui.status(f"[llm] {target} answered in {time.time() - started:.0f}s "
                   f"({len(out.split())} words)")
            return out
        except Exception as e:  # noqa: BLE001 - that is what the next link is for
            errors.append(f"{backend}/{target}: {type(e).__name__}: {e}")
            if i + 1 < len(CHAIN):
                ui.log(f"[error] {backend} not answering for {target}; moving on to "
                      f"{CHAIN[i + 1]}")
    failure = RuntimeError(f"no provider answered for {model} — "
                           + " | ".join(errors or ["empty chain"]))
    if INTERACTIVE and threading.current_thread() is threading.main_thread():
        # The caller asked for a role, not a literal id: if the recovery
        # repointed PRO/FLASH, the retry must chase the new ones instead of
        # handing the old name to a provider that has never heard of it
        # (opus is not in oauth's catalogue, so the new head skipped it).
        was_pro, was_flash = model == PRO, model == FLASH
        _recover()
        model = PRO if was_pro else FLASH if was_flash else model
        return chat(model, prompt, system, temperature=temperature,
                    max_tokens=max_tokens, retries=retries, fallback=fallback)
    raise failure


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def chat_json(model: str, prompt: str, system: str | None = None,
              **kw: Any) -> Any:
    """Completion parsed as JSON. Retries once with a repair nudge.

    Models wrap JSON in prose or fences often enough that a tolerant extractor
    beats strict parsing; the second pass only fires when extraction fails.
    """
    sys_msg = (system or "") + "\n\nAnswer ONLY with valid JSON, no extra text."
    raw = chat(model, prompt, sys_msg.strip(), **kw)
    for candidate in _json_candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    repaired = chat(model, f"Convert this into valid JSON and nothing else:\n\n{raw[:20000]}",
                    "Return only JSON.", temperature=0.0)
    for candidate in _json_candidates(repaired):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from {model}:\n{raw[:800]}")


def _json_candidates(raw: str):
    yield raw.strip()
    m = _FENCE.search(raw)
    if m:
        yield m.group(1).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = raw.find(open_c), raw.rfind(close_c)
        if i != -1 and j > i:
            yield raw[i:j + 1]
