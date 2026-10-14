from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

OLD_PROMPT_SENTENCE = "If you genuinely do not know, say 'I don't know' rather than inventing an answer."
NEW_PROMPT_SENTENCE = (
    "For common, well-known facts, answer directly from your learned knowledge. "
    "Do not reflexively answer 'I don't know' for established facts. "
    "If a question is genuinely ambiguous or lacks enough information, state uncertainty briefly without inventing details."
)


def replace_required(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new), True
    if new in text:
        return text, False
    raise RuntimeError(f"Expected code pattern not found: {label}")


def patch_gauge(app: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    pattern = re.compile(
        r"def gauge_svg\(probability: float, threshold: float, decision: str\) -> str:\n.*?(?=\n\ndef result_html\()",
        flags=re.S,
    )
    replacement = '''def gauge_svg(
    probability: float,
    threshold: float,
    decision: str,
    quality_ok: bool = True,
) -> str:
    """Draw the reliability gauge, hiding the score for a rejected non-answer."""
    cx = cy = 140
    radius = 100
    circumference = 2 * math.pi * radius
    plotted_probability = probability if quality_ok else 0.0
    probability_label = f"{probability * 100:.1f}%" if quality_ok else "N/A"
    aria_reliability = (
        f"Estimated reliability {probability * 100:.1f} percent"
        if quality_ok
        else "Reliability estimate unavailable because the response quality check failed"
    )
    offset = circumference * (1 - plotted_probability)

    zone_radius = radius + 9
    zone_circ = 2 * math.pi * zone_radius
    zone_dash = zone_circ * (1 - threshold)
    zone_start = zone_circ * threshold

    angle = 2 * math.pi * threshold

    def point(r: float) -> tuple[float, float]:
        return cx + r * math.sin(angle), cy - r * math.cos(angle)

    x1, y1 = point(radius - 15)
    x2, y2 = point(radius + 17)
    lx, ly = point(radius + 33)

    if decision == "TRUST":
        stop_a, stop_b, word_color = "#34d399", "#06b6d4", "#059669"
    else:
        stop_a, stop_b, word_color = "#fcd34d", "#f97316", "#b45309"

    return f"""
    <svg class="gauge" viewBox="0 0 280 280" role="img"
         aria-label="{aria_reliability}, trust boundary {pct_plain(threshold)} percent">
        <defs>
            <linearGradient id="ringgrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="{stop_a}"/>
                <stop offset="1" stop-color="{stop_b}"/>
            </linearGradient>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="#ffffff" stroke="#e8ecfb" stroke-width="18"/>
        <circle cx="{cx}" cy="{cy}" r="{zone_radius}" fill="none" stroke="#10b981" stroke-opacity="0.4"
                stroke-width="5" stroke-linecap="round" stroke-dasharray="{zone_dash:.2f} {zone_circ:.2f}"
                stroke-dashoffset="{-zone_start:.2f}" transform="rotate(-90 {cx} {cy})"/>
        <circle class="ring-prog" cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="url(#ringgrad)"
                stroke-width="18" stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}" style="--circ:{circumference:.2f};--off:{offset:.2f}"
                transform="rotate(-90 {cx} {cy})"/>
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#0f172a" stroke-width="3" stroke-linecap="round"/>
        <text x="{lx:.2f}" y="{ly + 4:.2f}" text-anchor="middle" font-size="12" font-weight="700" fill="#475569"
              style="font-family:inherit">{pct_plain(threshold)}%</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="50" font-weight="800" fill="#0f172a"
              style="font-family:inherit">{probability_label}</text>
        <text x="{cx}" y="{cy + 40}" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="#64748b"
              style="font-family:inherit">{"EST. RELIABILITY" if quality_ok else "NOT SCORED"}</text>
        <text x="{cx}" y="{cy + 66}" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="2" fill="{word_color}"
              style="font-family:inherit">{decision}</text>
    </svg>
    """
'''
    if pattern.search(app):
        app = pattern.sub(replacement, app, count=1)
        changes.append("Main reliability gauge now shows N/A when response quality fails")
    elif re.search(r"def gauge_svg\([\s\S]*?quality_ok: bool = True", app):
        changes.append("Main reliability gauge was already patched")
    else:
        raise RuntimeError("Could not safely locate gauge_svg() in app.py")

    app, changed = replace_required(
        app,
        "{gauge_svg(probability, threshold, decision)}",
        '{gauge_svg(probability, threshold, decision, quality_ok=view["quality_ok"])}',
        "gauge_svg call in result_html",
    )
    if changed:
        changes.append("Gauge rendering now receives the response-quality status")

    old_meter = 'style="--w:{probability * 100:.2f}%"></div>'
    new_meter = 'style="--w:{(probability * 100 if view[\'quality_ok\'] else 0.0):.2f}%"></div>'
    app, changed = replace_required(app, old_meter, new_meter, "meter fill width")
    if changed:
        changes.append("Progress meter no longer visualizes a rejected answer's score")

    old_disclaimer = """<div class="disclaimer">
                    Reliability is the gate's calibrated estimate of the probability that
                    this answer is correct. It is a decision policy, not a guarantee of factual truth.
                </div>"""
    new_disclaimer = """<div class="disclaimer">
                    {("No reliability score is shown because the answer failed the response-quality check."
                      if not view["quality_ok"] else
                      "Reliability is the gate's calibrated estimate of the probability that this answer is correct. It is a decision policy, not a guarantee of factual truth.")}
                </div>"""
    app, changed = replace_required(app, old_disclaimer, new_disclaimer, "result disclaimer")
    if changed:
        changes.append("Disclaimer now explains why a flagged response has no score")

    app, changed = replace_required(
        app,
        '("Raw P(correct)", f"{view[\'probability\']:.4f}"),',
        '("Raw classifier score (diagnostic only)", f"{view[\'probability\']:.4f}"),',
        "technical diagnostic score label",
    )
    if changed:
        changes.append("Technical panel labels the raw classifier score as diagnostic only")

    old_history_probability = "<b>{item['probability'] * 100:.0f}%</b>"
    new_history_probability = "<b>{'N/A' if item.get('quality_ok') is False else f\"{item['probability'] * 100:.0f}%\"}</b>"
    app, changed = replace_required(
        app, old_history_probability, new_history_probability, "history score display"
    )
    if changed:
        changes.append("History hides scores for responses that failed quality checks")

    old_history_store = '"probability": new_view["probability"],'
    new_history_store = '"probability": new_view["probability"],\n                        "quality_ok": new_view["quality_ok"],'
    app, changed = replace_required(
        app, old_history_store, new_history_store, "history quality flag storage"
    )
    if changed:
        changes.append("History stores response-quality status for correct score display")

    return app, changes


def patch_prompt(root: Path) -> tuple[dict[Path, str], list[str]]:
    candidates = [root / "src" / "gate" / "config.py", root / "src" / "gate" / "generation.py"]
    prompt_path = None
    prompt_text = None
    for path in candidates:
        if path.exists() and OLD_PROMPT_SENTENCE in path.read_text(encoding="utf-8"):
            prompt_path = path
            prompt_text = path.read_text(encoding="utf-8")
            break

    if prompt_path is None or prompt_text is None:
        return {}, [
            "PROMPT NOT CHANGED: could not find the exact V4 abstention sentence in config.py or generation.py. "
            "No guessing was used; send the current src/gate/config.py and src/gate/generation.py if this happens."
        ]

    updated = prompt_text.replace(OLD_PROMPT_SENTENCE, NEW_PROMPT_SENTENCE)
    changes = [f"Prompt updated in {prompt_path.relative_to(root)} to reduce reflexive abstention"]

    config_path = root / "src" / "gate" / "config.py"
    updated_files: dict[Path, str] = {prompt_path: updated}
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        if "qa-useful-v4" in config_text:
            config_text = config_text.replace("qa-useful-v4", "qa-useful-v5")
            updated_files[config_path] = config_text
            changes.append("Prompt version bumped from qa-useful-v4 to qa-useful-v5")
        elif re.search(r"PROMPT_VERSION\s*[:=].*['\"]", config_text):
            config_text2, count = re.subn(
                r"(PROMPT_VERSION\s*[:=]\s*['\"])[^'\"]*(['\"])",
                r"\1qa-useful-v5\2",
                config_text,
                count=1,
            )
            if count:
                updated_files[config_path] = config_text2
                changes.append("Prompt version bumped to qa-useful-v5")
        else:
            changes.append("WARNING: prompt text changed but a PROMPT_VERSION field was not recognized in config.py")

    return updated_files, changes


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the Local LLM Reliability Gate V5 prompt/UI correction.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (defaults to current directory)")
    args = parser.parse_args()
    root = args.root.resolve()

    app_path = root / "app.py"
    if not app_path.exists():
        print(f"ERROR: app.py not found in {root}")
        return 2
    if not (root / "src" / "gate").is_dir():
        print(f"ERROR: src/gate directory not found in {root}")
        return 2

    app_original = app_path.read_text(encoding="utf-8")
    app_updated, ui_changes = patch_gauge(app_original)

    prompt_files, prompt_changes = patch_prompt(root)
    all_updated: dict[Path, str] = {app_path: app_updated, **prompt_files}

    # Validate every modified Python file before writing anything.
    for path, content in all_updated.items():
        compile(content, str(path), "exec")

    changed_files = [path for path, content in all_updated.items() if not path.exists() or path.read_text(encoding="utf-8") != content]
    if not changed_files:
        print("No file contents changed. The UI may already contain this fix.")
        for item in prompt_changes:
            print(f"- {item}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "backups" / f"pre_v5_fix_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in changed_files:
        rel = path.relative_to(root)
        target = backup_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for path in changed_files:
        path.write_text(all_updated[path], encoding="utf-8", newline="\n")

    print("\nV5 FIX APPLIED")
    print(f"Project: {root}")
    print(f"Backup:  {backup_dir}")
    for item in ui_changes + prompt_changes:
        print(f"- {item}")
    print("\nIMPORTANT: a prompt/config change makes the V4 calibration stale. Do not call scores calibrated until the V5 benchmark and finalizer finish.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: no files were written: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
