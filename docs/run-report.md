# The run report

Every run writes `logs/<shot>_v###.run_report.json`: one file that says what happened, for a person, the GUI, the
CLI and an agent. It is written whatever the outcome (done, failed, cancelled).

```
<shot>_v### done: 4 ran, 1 fell back (ltx_alpha_refiner -> vitmatte); camera_pycolmap GOOD; 1 needs a human; <time>, VRAM <n> GB, RAM <n> GB
```

That line is `summary`; the CLI prints it at the end of `run-shot` and the GUI shows it in the log, the status bar
and the shot's tooltip (the row gets a flag with the count of items that need a human).

## What is in it

| Field | Meaning |
|---|---|
| `schema_version` | `1.0`. Fields are only added within a major version; a reader checks the major number. |
| `status`, `error` | `done` / `failed` / `cancelled`, and the error of a failed run. |
| `passes[]` | One entry per pass: `status` (`ran`, `fell_back`, `skipped`, `failed`, `not_run`), `reason`, `engine` (what actually ran: `sam3.1`, `sam3`, `ltx`, `vitmatte`, `pycolmap/depth_pnp` ...), `fallbacks[]` (wanted, used, why, which objects), `seconds`, `peak_vram_gb`, `peak_ram_gb`, `frames`, `channels`, `objects[]`, `metrics`, `checks[]`, `verdict` (GOOD / WEAK / FAILED), `warnings`, `license` / `commercial`. |
| `objects[]` | Per object a pass found: `id`, `name`, first / last frame, frames present, `area` (share of the frame, one value per frame). |
| `summaries` | Blocks a pass puts at the top: `camera` (solver, verdict, held-out / reprojection error and the resolution they are measured at, frames without a pose), `splat` (held-out PSNR, worst frame, masked share). |
| `needs_human[]` | `source`, `kind`, `reason`, `frames`: what a person should look at before using the run. |
| `environment` | Tool version, Python, torch / CUDA / GPU, a hash of the run's parameters. |
| `thresholds` | The limits and policy the run was judged with, and where they came from. |
| `notes` | What the numbers measure (e.g. VRAM is torch in this process: SAM 3.1 and COLMAP run in child processes, LTX in ComfyUI). |

`needs_human` kinds: `run_failed`, `pass_failed`, `pass_skipped`, `verdict` (a FAILED or WEAK pass), `fallback` (a
pass did not run the engine it was asked for), `non_commercial`, `matte_collapse` (an object lost or shrunk to a
sliver on frames X-Y), `camera_gap` (frames without a camera pose), `concept_not_found`.

## Verdicts and thresholds

A pass that judges itself (the camera's GOOD / WEAK / FAILED) keeps its own verdict. Everything else is judged by
`live_action_aov.core.verdicts`: named limits per metric and a policy for `needs_human`. The defaults are generic.
To use other values without touching code, give a JSON file in `LAOV_REPORT_THRESHOLDS`, or set
`Job.report_thresholds` (same format, wins over the file). The shape (the values are placeholders):

```json
{
  "limits": {"<metric>": {"good": 1.0, "weak": 2.0}, "<another metric>": null},
  "policy": {"weak_needs_human": false}
}
```

`null` switches a check off. The report's `thresholds` block lists the limits and policy keys in use.

## For pass authors

Override `UtilityPass.report_fields()` and return a dict (see `core.run_report.PassContribution`): `engine`,
`fallbacks`, `objects`, `metrics` (a number, or `{"value": x, "frames": [a, b]}`), `verdict`, `warnings`,
`needs_human`, `section` + `summary`, `extra`. It is called after `run_shot` (also after a failure) and before
`unload`. The executor records status, time, VRAM, RAM and channels by itself. A bad dict is a warning in the
report, never a failed run.
