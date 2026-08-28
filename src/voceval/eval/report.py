from __future__ import annotations

import json
from html import escape

from voceval.eval.runner import ScenarioResult


def to_json(results: list[ScenarioResult]) -> dict:
    return {
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "scenarios": [
            {
                "name": r.scenario,
                "passed": r.passed,
                "metrics": r.metrics.to_dict(),
                "scores": [
                    {"name": s.name, "passed": s.passed, "detail": s.detail}
                    for s in r.scores
                ],
                "transcript": r.dialogue.transcript(),
            }
            for r in results
        ],
    }


def to_markdown(results: list[ScenarioResult]) -> str:
    passed = sum(r.passed for r in results)
    lines = [
        f"# Eval run: {passed}/{len(results)} scenarios passed",
        "",
        "| scenario | result | p50 | p95 | interruption |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r.metrics
        mark = "pass" if r.passed else "**fail**"
        lines.append(
            f"| {r.scenario} | {mark} | {m.p50_response_latency:.2f}s | "
            f"{m.p95_response_latency:.2f}s | {m.max_interruption_response:.2f}s |"
        )

    for r in results:
        if r.passed:
            continue
        lines += ["", f"## {r.scenario}", ""]
        for s in r.failures():
            lines.append(f"- {s.name}: {s.detail or 'failed'}")
        lines += ["", "```", r.dialogue.transcript(), "```"]

    return "\n".join(lines) + "\n"


def to_html(results: list[ScenarioResult]) -> str:
    rows = ""
    for r in results:
        m = r.metrics
        cls = "pass" if r.passed else "fail"
        checks = "".join(
            f"<li class='{'ok' if s.passed else 'bad'}'>{escape(s.name)} "
            f"<span>{escape(s.detail)}</span></li>"
            for s in r.scores
        )
        rows += f"""
        <section class="{cls}">
          <h2>{escape(r.scenario)}</h2>
          <p>p50 {m.p50_response_latency:.2f}s &middot; p95 {m.p95_response_latency:.2f}s
             &middot; interruption {m.max_interruption_response:.2f}s
             &middot; talk-over {m.talk_over_ratio:.0%}</p>
          <ul>{checks}</ul>
          <pre>{escape(r.dialogue.transcript())}</pre>
        </section>"""

    passed = sum(r.passed for r in results)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>voceval report</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 820px; }}
  section {{ border-left: 4px solid #999; padding: 0 1rem; margin: 1rem 0; }}
  section.pass {{ border-color: #2e7d32; }}
  section.fail {{ border-color: #c62828; }}
  li.bad {{ color: #c62828; }}
  li span {{ color: #666; }}
  pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; }}
</style>
<h1>{passed}/{len(results)} scenarios passed</h1>
{rows}
"""


def write_reports(results: list[ScenarioResult], directory: str) -> None:
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(to_json(results), indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(results), encoding="utf-8")
    (out / "report.html").write_text(to_html(results), encoding="utf-8")
