"""Terminal colouring for the CLI.

ANSI escapes only — no dependency, and everything degrades to plain text when
stdout is not a terminal (a redirected log stays readable) or NO_COLOR is set.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# Old Windows consoles need VT processing switched on; one no-op os.system call
# does it. Windows Terminal already has it and does not care.
if os.name == "nt":
    os.system("")

ENABLED = bool(getattr(sys.stdout, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")

# One colour per pipeline stage, so a long run reads as bands instead of a wall.
TAGS = {
    "tema": MAGENTA,
    "investigación": BLUE,
    "bibliografía": BLUE,
    "esquema": CYAN,
    "borrador": CYAN,
    "revisión": YELLOW,
    "detector": YELLOW,
    "aprobación": GREEN,
    "publicación": GREEN,
    "continuo": MAGENTA,
    "retomar": DIM,
    "error": RED,
}

_TAG = re.compile(r"^(\s*)\[([^\]]+)\]")


def c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, or return it untouched when colour is off."""
    return f"{''.join(codes)}{text}{RESET}" if ENABLED else text


def rule(title: str) -> str:
    return c(f"═══ {title} ═══", BOLD, CYAN)


def _line(line: str) -> str:
    m = _TAG.match(line)
    if m:
        colour = TAGS.get(m.group(2).split()[0].lower(), CYAN)
        return f"{m.group(1)}{c('[' + m.group(2) + ']', BOLD, colour)}{line[m.end():]}"
    stripped = line.strip()
    if stripped and not stripped.strip("─═-="):  # box-drawing separators
        return c(line, CYAN)
    if stripped.startswith(("·", "-")):
        return c(line, DIM)
    return line


_LAST_TAG: str | None = None

# Auto runs (set by main): model-call lines, heartbeats, «· detail» bullets and
# cache hits stop scrolling and share one status line redrawn in place. Stage
# lines ([tema], [borrador]…) and errors still print. Off = the full log.
COMPACT = False
TTY = bool(getattr(sys.stdout, "isatty", lambda: False)())
_LOCK = threading.RLock()
_STATUS = False


def _chatter(msg: str) -> bool:
    first = msg.lstrip("\n").split("\n", 1)[0]
    if "] [#" in first or "] [." in first:  # progress bars stay in the log
        return False
    return (first.lstrip().startswith("·") or first.startswith("[llm]")
            or "reutilizando" in first or "en caché" in first
            or first.rstrip().endswith("…")  # «doing X…» announcements
            or (first[:1] == " " and not first.lstrip().startswith("[")))  # indented detail


def _clear_status() -> None:
    global _STATUS
    if _STATUS:
        sys.stdout.write("\r" + " " * (shutil.get_terminal_size().columns - 1) + "\r")
        _STATUS = False


def status(msg: str) -> None:
    """Transient line: redrawn in place when compact, a normal log line otherwise."""
    global _STATUS
    if not COMPACT:
        log(msg)
        return
    if not TTY:  # a redirected log gets no carriage-return noise
        return
    width = shutil.get_terminal_size().columns - 1
    line = " ".join(msg.split())
    with _LOCK:
        sys.stdout.write("\r" + line[:width].ljust(width))
        sys.stdout.flush()
        _STATUS = True


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = width * done // max(1, total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


def log(msg: str = "") -> None:
    """print() with the [etapa] prefix, separators and bullets coloured.

    A blank line separates each change of stage tag, so a long run reads as
    blocks instead of a wall of text.
    """
    if COMPACT and _chatter(str(msg)):
        status(str(msg))
        return
    with _LOCK:
        _clear_status()
        _log(msg)


def _log(msg: str = "") -> None:
    global _LAST_TAG
    m = _TAG.match(str(msg))
    if m:
        tag = m.group(2).split()[0].lower()
        if _LAST_TAG is not None and tag != _LAST_TAG and not COMPACT:
            print()
        _LAST_TAG = tag
    if not ENABLED:
        print(msg)
        return
    print("\n".join(_line(l) for l in str(msg).split("\n")))


def demo() -> None:
    assert _line("hola") == "hola"
    if not ENABLED:  # nothing else is observable with colour off
        return
    assert _line("[tema] x") == c("[tema]", BOLD, MAGENTA) + " x"
    assert _line("  [detector] y").startswith("  ")
    assert _line("[loquesea] z").startswith(BOLD + CYAN)
    assert _line("  · nota").startswith(DIM)
    assert _line("─────").startswith(CYAN)
    log("[tema] «Un título»\n  · una nota\n─────\n[error] algo falló")
    assert progress_bar(2, 4, 4) == "[##..] 2/4"
    assert _chatter("  · juez x: 80% IA") and _chatter("[llm] a → b")
    assert _chatter("[borrador] intro en caché (300p)")
    assert _chatter("[esquema] auditoría del esquema por PRO…")
    assert _chatter("  hipótesis: algo")
    assert not _chatter("[borrador] [##..] 1/3 intro (500p)…")
    assert not _chatter("[revisión] veredicto: ok (80/100)")


if __name__ == "__main__":
    demo()
