from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voceval.config import Settings
from voceval.eval.loader import load_agent
from voceval.eval.regression import compare, save_baseline
from voceval.eval.report import to_markdown, write_reports
from voceval.eval.runner import run_scenario, run_suite
from voceval.eval.scenario import load_scenario, load_suite

app = typer.Typer(add_completion=False, help="Voice agent runtime and evaluation.")
console = Console()


@app.command()
def chat(agent: str = typer.Option(..., help="Path to an agent entrypoint")) -> None:
    """Talk to an agent from the terminal (types stand in for speech)."""
    from voceval.eval.simulated import SimulatedChannel

    settings = Settings.load()
    orchestrator = load_agent(agent, settings).orchestrator()

    async def loop() -> None:
        channel = SimulatedChannel(settings.sample_rate)
        runner = asyncio.create_task(orchestrator.run(channel))
        seen = 0
        while True:
            await asyncio.sleep(0.1)
            if orchestrator.is_responding:
                continue
            for turn in orchestrator.turns[seen:]:
                console.print(f"[bold]{turn.speaker.value}[/]: {turn.text}")
            seen = len(orchestrator.turns)
            try:
                line = console.input("[dim]you > [/]")
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                break
            channel.mark_turn()
            await channel.caller_says(line)
        await channel.close()
        runner.cancel()

    asyncio.run(loop())


@app.command()
def simulate(
    scenario: str = typer.Option(..., help="Path to a scenario YAML"),
    report: str = typer.Option("", help="Directory to write reports into"),
) -> None:
    """Run one scenario and show the transcript and scores."""
    settings = Settings.load()
    result = asyncio.run(run_scenario(load_scenario(scenario), settings))

    console.print(f"\n[bold]{result.scenario}[/] "
                  + ("[green]passed[/]" if result.passed else "[red]failed[/]"))
    console.print(result.dialogue.transcript(), style="dim")

    table = Table(show_header=True)
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail")
    for s in result.scores:
        table.add_row(s.name, "ok" if s.passed else "fail", s.detail)
    console.print(table)
    console.print(result.metrics.to_dict())

    if report:
        write_reports([result], report)
        console.print(f"reports written to {report}")
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def eval(
    suite: str = typer.Option(..., help="Directory of scenario YAML files"),
    baseline: str = typer.Option("", help="Baseline JSON to compare against"),
    report: str = typer.Option("reports", help="Directory to write reports into"),
    update_baseline: bool = typer.Option(False, help="Overwrite the baseline with this run"),
) -> None:
    """Run a scenario suite, write reports, and check for regressions."""
    settings = Settings.load()
    scenarios = load_suite(suite)
    if not scenarios:
        console.print(f"[red]no scenarios found in {suite}[/]")
        raise typer.Exit(2)

    results = asyncio.run(run_suite(scenarios, settings))
    console.print(to_markdown(results))
    write_reports(results, report)

    if update_baseline:
        target = baseline or str(Path(".voceval") / "baseline.json")
        save_baseline(results, target)
        console.print(f"baseline written to {target}")
        return

    failed = [r for r in results if not r.passed]
    regressions = compare(baseline, results) if baseline and Path(baseline).exists() else []

    for reg in regressions:
        console.print(f"[red]regression[/] {reg.scenario}: {reg.kind} — {reg.detail}")

    if failed or regressions:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
