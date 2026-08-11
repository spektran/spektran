# AI Agent Guide

SPEKTRAN is fully **AI Agent-ready**. The entire ML pipeline — simulate, generate,
train, evaluate — is operable via CLI commands with structured JSON output.

!!! tip "For AI agents"
    Read [`AGENTS.md`](https://github.com/spektran/spektran/blob/main/AGENTS.md) in
    the repo root for the complete machine-readable interface specification.

## How It Works

AI coding agents (Claude Code, Cursor, GitHub Copilot, Windsurf, Cline) can
operate SPEKTRAN by running shell commands. Every command supports `--json`
for structured output that agents can parse directly.

```
User: "Train all available baselines on T1 and compare their MAE"

Agent reads AGENTS.md → discovers commands → executes:
  spektran list baselines --json  → finds ridge, cnn1d, transformer support T1
  spektran train --baseline ridge --json → auto-generates data, trains, reports scores
  spektran train --baseline cnn1d --json → same
  spektran train --baseline transformer --json → same
  → presents comparison table to user
```

## Commands

### Discovery

| Command | What it returns |
|---------|-----------------|
| `spektran info --json` | Project overview: version, tasks, baselines, data status |
| `spektran list tasks --json` | All 9 tasks with input/target/metrics/available baselines |
| `spektran list baselines --json` | All 14 baselines with descriptions and scores |
| `spektran list instruments --json` | All 18 virtual instruments with tier and technique |
| `spektran list datasets --json` | All dataset configs with generation status |
| `spektran status --json` | Combined data + training state |

### Pipeline

| Command | What it does |
|---------|--------------|
| `spektran train --baseline <name> --json` | Auto-generates data + trains + reports scores |
| `spektran generate <config.yaml> --out data --json` | Generate a specific dataset |
| `spektran benchmark --task <id> --truth <h5> --predictions <csv>` | Evaluate predictions |

### Error Handling

All errors in `--json` mode return structured output:

```json
{"error": "Unknown baseline 'foo'. Available: ridge, cnn1d, ...", "code": 1}
```

Non-zero exit codes always indicate failure.

## Workflow Examples

### Train and evaluate a baseline

```bash
spektran train --baseline ridge --task T1 --json
```

Output:
```json
{
  "baseline": "ridge",
  "display_name": "Ridge Regression",
  "tasks_trained": ["T1"],
  "data_generated": [],
  "scores": {
    "T1": {"mae_ppm": 2.84, "mape_pct": 29.87, "rmse_ppm": 4.46}
  }
}
```

### Compare baselines

```bash
for b in ridge cnn1d transformer; do
  spektran train --baseline $b --task T1 --json 2>/dev/null | \
    python -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"baseline\"]:20s} MAE={d[\"scores\"][\"T1\"][\"scores\"][\"mae_ppm\"]:.2f}')"
done
```

### Custom model workflow

1. Generate data: `spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data --json`
2. Write your model (load data via `spektran.io.read_records`)
3. Write predictions CSV: `record_id,concentration_ppm`
4. Evaluate: `spektran benchmark --task T1-concentration --truth data/ch4-t1-test-v0.h5 --predictions my_preds.csv`

## Supported Agents

SPEKTRAN's CLI interface works with any agent that can execute shell commands:

- **Claude Code** — reads `AGENTS.md` automatically
- **Cursor** — reads project context files
- **GitHub Copilot** — operates via terminal
- **Windsurf** — reads project documentation
- **Cline** — operates via terminal
- **Any custom agent** — just parse `--json` output

## Design Principles

1. **CLI-first** — shell commands are the universal agent interface
2. **Discoverable** — `spektran info` returns everything an agent needs to plan
3. **JSON everywhere** — `--json` on every command for structured output
4. **Convention over configuration** — `spektran train --baseline ridge` just works
5. **Idempotent** — safe to re-run; data generation skips existing files
6. **Errors as data** — `--json` mode returns structured errors, not stderr text
