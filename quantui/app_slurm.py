"""
SLURM execution path for QuantUIApp (M-CLUSTER2 CL2.3).

Submits batch jobs via :class:`~quantui.backends.slurm.SlurmBackend`, polls the
on-disk registry + staging logs, and ingests finished single-point results into
History. Local in-kernel runs remain the default.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from IPython.display import HTML

from quantui import theme as _theme
from quantui.backends.dispatch import (
    build_calculation_request,
    calc_type_key_from_app,
    is_slurm_available,
    slurm_backend_for_app,
)
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm_errors import format_error_html

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_SUPPORTED_SLURM_CALC_TYPES = frozenset({"single_point"})


def ensure_job_registry(app: Any) -> JobRegistry:
    registry = getattr(app, "_job_registry", None)
    if registry is None:
        registry = JobRegistry()
        app._job_registry = registry
    return registry


def startup_slurm_check(app: Any) -> None:
    """On app load, surface any in-flight SLURM jobs from the registry."""
    if not is_slurm_available():
        return
    ensure_job_registry(app)
    active = [
        r for r in app._job_registry.list_active() if r.backend_id == "cluster_slurm"
    ]
    if not active:
        return
    record = active[0]
    _show_slurm_banner(
        app,
        f"SLURM job still active (status: {record.status}). "
        f"Click <b>View progress</b> to reconnect.",
        request_id=record.request_id,
    )


def submit_slurm_run(app: Any) -> None:
    """Submit the current configuration to SLURM and start the monitor thread."""
    if not is_slurm_available():
        app.run_status.value = (
            "SLURM is not available on this system (sbatch not found)."
        )
        return

    calc_type = calc_type_key_from_app(app)
    if calc_type not in _SUPPORTED_SLURM_CALC_TYPES:
        app.run_status.value = (
            f"SLURM batch mode currently supports Single Point only "
            f"(selected: {app.calc_type_dd.value}). Switch to Local or Single Point."
        )
        return

    ensure_job_registry(app)
    backend = slurm_backend_for_app(app)
    request = build_calculation_request(app)

    app._calc_running = True
    app._slurm_monitor_stop = threading.Event()
    app.cancel_btn.disabled = False
    app.log_clear_btn.disabled = True
    app.run_btn.disabled = True
    app.run_status.value = "Submitting to SLURM…"

    try:
        request_id = backend.dispatch(request)
    except RuntimeError as exc:
        app._calc_running = False
        app.run_btn.disabled = False
        app.cancel_btn.disabled = True
        app.log_clear_btn.disabled = False
        app.run_status.value = "SLURM submission failed."
        _append_run_html(app, format_error_html(str(exc)))
        return

    app._slurm_active_request_id = request_id
    record = app._job_registry.load(request_id)
    slurm_id = record.slurm_job_id if record else "?"
    app.run_status.value = f"Submitted to SLURM (job {slurm_id}). Monitoring…"
    _append_run_stdout(
        app,
        f"\n📤 Submitted batch job {slurm_id} (request {request_id})\n"
        f"Staging directory: {record.staging_dir if record else '?'}\n",
    )
    threading.Thread(
        target=_monitor_slurm_job,
        args=(app, request_id),
        daemon=True,
        name=f"quantui-slurm-monitor-{request_id}",
    ).start()


def cancel_slurm_run(app: Any) -> bool:
    """Cancel the active SLURM job if one is being monitored."""
    request_id = getattr(app, "_slurm_active_request_id", None)
    if not request_id:
        return False
    stop = getattr(app, "_slurm_monitor_stop", None)
    if stop is not None:
        stop.set()
    backend = slurm_backend_for_app(app)
    ok = backend.cancel(request_id)
    app.run_status.value = "SLURM cancellation requested."
    return ok


def attach_slurm_job(app: Any, request_id: str) -> None:
    """Reconnect the UI to an existing SLURM job (browser reopen path)."""
    ensure_job_registry(app)
    record = app._job_registry.load(request_id)
    if record is None:
        app.run_status.value = "SLURM job record not found."
        return
    if record.status in ("success", "error", "cancelled"):
        _ingest_terminal_job(app, record)
        return

    app._calc_running = True
    app._slurm_active_request_id = request_id
    app._slurm_monitor_stop = threading.Event()
    app.cancel_btn.disabled = False
    app.log_clear_btn.disabled = True
    app.run_btn.disabled = True
    app.run_status.value = (
        f"Reconnected to SLURM job {record.slurm_job_id or request_id}."
    )
    _hide_slurm_banner(app)
    threading.Thread(
        target=_monitor_slurm_job,
        args=(app, request_id),
        daemon=True,
        name=f"quantui-slurm-monitor-{request_id}",
    ).start()


def _monitor_slurm_job(app: Any, request_id: str) -> None:
    backend = slurm_backend_for_app(app)
    stop = getattr(app, "_slurm_monitor_stop", threading.Event())
    last_log_size = 0

    while not stop.is_set():
        backend.refresh_registry_statuses()
        record = app._job_registry.load(request_id)
        if record is None:
            break

        log_path = record.live_log_path
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > last_log_size:
                    chunk = text[last_log_size:]
                    last_log_size = len(text)
                    app._queue_main_thread_callback(_append_run_stdout, app, chunk)
            except OSError:
                pass

        progress_path = record.progress_path
        if progress_path.exists():
            try:
                prog = json.loads(progress_path.read_text(encoding="utf-8"))
                msg = prog.get("message", "")
                pct = prog.get("percent")
                if msg:
                    status = f"SLURM: {msg}"
                    if pct is not None:
                        status += f" ({pct:.0f}%)"
                    app._queue_main_thread_callback(_set_run_status, app, status)
            except (OSError, json.JSONDecodeError):
                pass

        if record.status in ("success", "error", "cancelled"):
            app._queue_main_thread_callback(_ingest_terminal_job, app, record)
            break

        slurm_id = record.slurm_job_id or "?"
        status_line = f"SLURM job {slurm_id}: {record.status}"
        app._queue_main_thread_callback(_set_run_status, app, status_line)

        time.sleep(_POLL_INTERVAL_S)

    app._queue_main_thread_callback(_finish_slurm_monitor, app)


def _ingest_terminal_job(app: Any, record: Any) -> None:
    if record.status == "success":
        _ingest_success(app, record)
    elif record.status == "cancelled":
        app.run_status.value = "SLURM job cancelled."
        _append_run_stdout(app, "\n⏹ SLURM job cancelled.\n")
    else:
        err = (record.error or {}).get("user_message") or "SLURM job failed."
        app.run_status.value = "SLURM job failed."
        _append_run_html(app, format_error_html(err))


def _ingest_success(app: Any, record: Any) -> None:
    staging = record.staging_path
    result_path = staging / "result.json"
    log_path = record.live_log_path
    if not result_path.exists():
        app.run_status.value = "SLURM job finished but result.json is missing."
        return

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )

    from quantui.session_calc import SessionResult

    result = SessionResult(
        energy_hartree=float(payload["energy_hartree"]),
        homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
        converged=bool(payload.get("converged", False)),
        n_iterations=int(payload.get("n_iterations", -1)),
        method=str(payload.get("method", record.request_obj.method)),
        basis=str(payload.get("basis", record.request_obj.basis)),
        formula=str(payload.get("formula", "?")),
    )

    try:
        from quantui import save_result
        from quantui.app_runflow import refresh_results_browser

        saved_dir = save_result(
            result,
            pyscf_log=log_text,
            calc_type="single_point",
        )
        app._job_registry.update_status(
            record.request_id, "success", result_dir=str(saved_dir)
        )
        app._last_result_dir = saved_dir
        refresh_results_browser(app)
        app.run_status.value = f"SLURM job complete — saved to {saved_dir.name}."
        _append_run_stdout(app, f"\n✅ Result saved to {saved_dir}\n")
        app.result_output.append_display_data(
            HTML(
                f'<div style="padding:12px;background:#ecfdf5;border-radius:8px;">'
                f"<b>SLURM calculation complete</b><br>"
                f"Energy: {result.energy_hartree:.6f} Ha<br>"
                f"Saved: <code>{saved_dir}</code></div>"
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface save failures
        logger.exception("Failed to ingest SLURM result")
        app.run_status.value = f"SLURM finished but save failed: {exc}"


def _finish_slurm_monitor(app: Any) -> None:
    app._calc_running = False
    app._slurm_active_request_id = None
    app.run_btn.disabled = False
    app.cancel_btn.disabled = True
    app.log_clear_btn.disabled = False
    _hide_slurm_banner(app)
    stop = getattr(app, "_slurm_monitor_stop", None)
    if stop is not None:
        stop.clear()


def _show_slurm_banner(app: Any, message: str, *, request_id: str) -> None:
    banner = getattr(app, "_slurm_job_banner", None)
    if banner is None:
        return
    banner.value = (
        f'<div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;'
        f'padding:10px 12px;margin:6px 0;font-size:13px;color:{_theme.TEXT_STRONG}">'
        f"ℹ️ {message}</div>"
    )
    banner.layout.display = ""
    btn = getattr(app, "_slurm_reconnect_btn", None)
    if btn is not None:
        btn.layout.display = ""
        btn._slurm_request_id = request_id  # type: ignore[attr-defined]


def _hide_slurm_banner(app: Any) -> None:
    banner = getattr(app, "_slurm_job_banner", None)
    if banner is not None:
        banner.layout.display = "none"
    btn = getattr(app, "_slurm_reconnect_btn", None)
    if btn is not None:
        btn.layout.display = "none"


def _set_run_status(app: Any, message: str) -> None:
    try:
        app.run_status.value = message
    except Exception:
        pass


def _append_run_stdout(app: Any, text: str) -> None:
    try:
        app.run_output.append_stdout(text)
    except Exception:
        pass


def _append_run_html(app: Any, html: str) -> None:
    try:
        app.run_output.append_display_data(HTML(html))
    except Exception:
        pass


def on_slurm_reconnect_clicked(app: Any, _btn: Any = None) -> None:
    btn = getattr(app, "_slurm_reconnect_btn", None)
    request_id = getattr(btn, "_slurm_request_id", None) if btn else None
    if request_id:
        attach_slurm_job(app, request_id)


def use_slurm_execution(app: Any) -> bool:
    pref = getattr(app._user_settings.compute, "execution_backend", "local")
    return pref == "slurm" and is_slurm_available()
