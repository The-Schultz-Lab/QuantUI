"""
SLURM execution path for QuantUIApp (M-CLUSTER2 CL2.3).

Submits batch jobs via :class:`~quantui.backends.slurm.SlurmBackend`, polls the
on-disk registry + staging logs, and ingests finished single-point results into
History. Local in-kernel runs remain the default.
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from typing import Any

from IPython.display import HTML

from quantui import theme as _theme
from quantui.backends import cluster_config as _cluster_cfg
from quantui.backends.base import CALC_TYPES
from quantui.backends.dispatch import (
    build_calculation_request,
    calc_type_key_from_app,
    is_slurm_available,
    slurm_backend_for_app,
    slurm_unavailable_user_message,
)
from quantui.backends.registry import JobRecord, JobRegistry
from quantui.backends.slurm_errors import format_error_html
from quantui.security import SecurityError

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_SUPPORTED_SLURM_CALC_TYPES = frozenset(CALC_TYPES)
_ACTIVE_SLURM_STATUSES = frozenset({"queued", "pending", "running", "submitted"})
_DISMISSABLE_SLURM_STATUSES = frozenset({"success", "error", "cancelled"})


def max_concurrent_slurm_jobs() -> int:
    return _cluster_cfg.max_concurrent_jobs()


def submit_cooldown_seconds() -> int:
    return _cluster_cfg.submit_cooldown_seconds()


def _reconcile_slurm_registry(app: Any) -> None:
    """Drop stale active rows before limit/cooldown checks or tab refresh."""
    if not is_slurm_available():
        return
    ensure_job_registry(app)
    try:
        slurm_backend_for_app(app).reconcile_stale_records()
    except Exception:  # noqa: BLE001 — reconciliation must not break the UI
        logger.exception("Failed to reconcile SLURM registry")


def list_slurm_jobs(app: Any) -> list[JobRecord]:
    ensure_job_registry(app)
    return [
        record
        for record in app._job_registry.list_all()
        if record.backend_id == "cluster_slurm"
    ]


def active_slurm_job_count(app: Any) -> int:
    return sum(
        1
        for record in list_slurm_jobs(app)
        if record.status.lower() in _ACTIVE_SLURM_STATUSES
    )


def slurm_submit_block_reason(app: Any) -> str | None:
    """Return a user-facing reason when a new SLURM submit should be blocked."""
    _reconcile_slurm_registry(app)

    cooldown = submit_cooldown_seconds()
    if cooldown > 0:
        registry = ensure_job_registry(app)
        since = registry.seconds_since_last_slurm_submit()
        if since is not None and since < cooldown:
            remaining = max(1, int(cooldown - since + 0.999))
            return f"Please wait {remaining}s before submitting another cluster job."

    limit = max_concurrent_slurm_jobs()
    active = active_slurm_job_count(app)
    if active >= limit:
        return (
            f"Concurrent job limit reached ({active}/{limit}). "
            "Open the Cluster Jobs tab to view running jobs or cancel one."
        )
    return None


def slurm_jobs_tab_visible(app: Any) -> bool:
    return use_slurm_execution(app)


def _reset_run_ui_after_submit_failure(app: Any) -> None:
    app._calc_running = False
    app.run_btn.disabled = False
    app.cancel_btn.disabled = True
    app.log_clear_btn.disabled = False


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
    _reconcile_slurm_registry(app)
    active = [
        r for r in app._job_registry.list_active() if r.backend_id == "cluster_slurm"
    ]
    if not active:
        return
    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)
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
        app.run_status.value = slurm_unavailable_user_message()
        return

    calc_type = calc_type_key_from_app(app)
    if calc_type not in _SUPPORTED_SLURM_CALC_TYPES:
        app.run_status.value = (
            f"SLURM batch mode does not support "
            f"{app.calc_type_dd.value!r} yet. Switch to Local."
        )
        return

    block_reason = slurm_submit_block_reason(app)
    if block_reason:
        app.run_status.value = block_reason
        _append_run_html(app, format_error_html(block_reason))
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
    except SecurityError as exc:
        _reset_run_ui_after_submit_failure(app)
        app.run_status.value = str(exc)
        _append_run_html(app, format_error_html(str(exc)))
        refresh_slurm_jobs_tab(app)
        _update_slurm_jobs_tab_title(app)
        return
    except RuntimeError as exc:
        _reset_run_ui_after_submit_failure(app)
        app.run_status.value = "SLURM submission failed."
        _append_run_html(app, format_error_html(str(exc)))
        return

    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)

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


def cancel_slurm_job(app: Any, request_id: str) -> bool:
    """Cancel a SLURM job by registry request id. Returns True when confirmed."""
    ensure_job_registry(app)
    backend = slurm_backend_for_app(app)
    if getattr(app, "_slurm_active_request_id", None) == request_id:
        stop = getattr(app, "_slurm_monitor_stop", None)
        if stop is not None:
            stop.set()
    ok = backend.cancel(request_id)
    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)
    return ok


def slurm_cancel_user_message(app: Any, request_id: str) -> str:
    """Return a user-facing cancel outcome message for *request_id*."""
    record = ensure_job_registry(app).load(request_id)
    if record is None:
        return "SLURM job record not found."
    if record.status == "cancelled":
        return f"SLURM job {request_id} cancelled."
    err = record.error or {}
    if err.get("user_message"):
        return str(err["user_message"])
    return f"Could not cancel SLURM job {request_id}."


def cancel_slurm_run(app: Any) -> bool:
    """Cancel the active SLURM job if one is being monitored."""
    request_id = getattr(app, "_slurm_active_request_id", None)
    if not request_id:
        return False
    ok = cancel_slurm_job(app, request_id)
    app.run_status.value = slurm_cancel_user_message(app, request_id)
    if ok:
        _finish_slurm_monitor(app)
    else:
        _append_run_html(app, format_error_html(app.run_status.value))
    return ok


def remove_slurm_job_record(app: Any, request_id: str) -> bool:
    """Delete a terminal registry row (does not remove staging files)."""
    registry = ensure_job_registry(app)
    record = registry.load(request_id)
    if record is None:
        return False
    if record.status.lower() in _ACTIVE_SLURM_STATUSES:
        return False
    if getattr(app, "_slurm_active_request_id", None) == request_id:
        stop = getattr(app, "_slurm_monitor_stop", None)
        if stop is not None:
            stop.set()
        _finish_slurm_monitor(app)
    return registry.delete(request_id)


def slurm_remove_user_message(app: Any, request_id: str, *, removed: bool) -> str:
    if removed:
        return f"Removed job {request_id} from your registry."
    record = ensure_job_registry(app).load(request_id)
    if record is None:
        return f"Job {request_id} is not in your registry."
    if record.status.lower() in _ACTIVE_SLURM_STATUSES:
        return (
            f"Job {request_id} is still active ({record.status}). "
            "Cancel it before removing."
        )
    return f"Could not remove job {request_id}."


def _format_slurm_job_option(record: JobRecord) -> tuple[str, str]:
    req = record.request_obj
    slurm_id = record.slurm_job_id or "pending"
    err = record.error or {}
    err_hint = f" | {err['code']}" if err.get("code") else ""
    label = (
        f"{record.status} | {slurm_id} | "
        f"{req.method}/{req.basis} | {record.request_id[:8]}{err_hint}"
    )
    return label, record.request_id


def _slurm_job_error_cell(record: JobRecord) -> str:
    err = record.error
    if not err:
        return "—"
    code = str(err.get("code") or "error")
    message = str(err.get("user_message") or err.get("technical_message") or code)
    if len(message) > 80:
        message = message[:77] + "…"
    return f"{html.escape(code)}: {html.escape(message)}"


def refresh_slurm_jobs_tab(app: Any) -> None:
    """Re-render the Cluster Jobs tab from the on-disk registry."""
    summary = getattr(app, "_slurm_jobs_summary_html", None)
    table = getattr(app, "_slurm_jobs_table_html", None)
    select = getattr(app, "_slurm_jobs_select", None)
    if summary is None or table is None or select is None:
        return

    if is_slurm_available():
        _reconcile_slurm_registry(app)
        backend = slurm_backend_for_app(app)
        try:
            backend.refresh_registry_statuses()
        except Exception:  # noqa: BLE001 — UI refresh must not crash the app
            logger.exception("Failed to refresh SLURM registry statuses")

    records = list_slurm_jobs(app)
    active = active_slurm_job_count(app)
    limit = max_concurrent_slurm_jobs()

    accounting: dict[str, Any] = {}
    if is_slurm_available():
        slurm_ids = [r.slurm_job_id for r in records if r.slurm_job_id]
        if slurm_ids:
            try:
                accounting = slurm_backend_for_app(app).batch_job_accounting(
                    slurm_ids  # type: ignore[arg-type]
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load SLURM job accounting")

    summary.value = (
        f'<div style="font-size:13px;color:{_theme.css.TEXT_STRONG};margin:4px 0 8px">'
        f"<b>{active}</b> active / <b>{limit}</b> max concurrent cluster job(s). "
        f"{len(records)} total in your registry.</div>"
    )

    if not records:
        table.value = (
            f'<div style="font-size:12px;color:{_theme.css.TEXT_SUBTLE};'
            f'padding:8px 0">No cluster jobs yet.</div>'
        )
        select.options = [("(no jobs)", "")]
        select.value = ""
        select.disabled = True
        return

    rows = []
    for record in records:
        req = record.request_obj
        slurm_id = record.slurm_job_id or "—"
        created = record.created_at.replace("T", " ").replace("+00:00", " UTC")
        acct = accounting.get(record.slurm_job_id or "")
        slurm_state = acct.state if acct else "—"
        elapsed = acct.elapsed if acct and acct.elapsed else "—"
        exit_code = acct.exit_code if acct and acct.exit_code else "—"
        rows.append(
            "<tr>"
            f"<td style='padding:4px 8px'>{record.status}</td>"
            f"<td style='padding:4px 8px'>{slurm_id}</td>"
            f"<td style='padding:4px 8px'>{html.escape(slurm_state)}</td>"
            f"<td style='padding:4px 8px'>{html.escape(elapsed)}</td>"
            f"<td style='padding:4px 8px'>{html.escape(exit_code)}</td>"
            f"<td style='padding:4px 8px'>{req.method}/{req.basis}</td>"
            f"<td style='padding:4px 8px'>{record.calc_type}</td>"
            f"<td style='padding:4px 8px'>{created}</td>"
            f"<td style='padding:4px 8px'>{_slurm_job_error_cell(record)}</td>"
            f"<td style='padding:4px 8px'><code>{record.request_id}</code></td>"
            "</tr>"
        )

    table.value = (
        f'<div style="overflow-x:auto;margin:4px 0 8px">'
        f'<table style="border-collapse:collapse;font-size:12px;width:100%">'
        f"<thead><tr style='background:{_theme.css.BG_PANEL}'>"
        "<th style='padding:4px 8px;text-align:left'>Status</th>"
        "<th style='padding:4px 8px;text-align:left'>SLURM ID</th>"
        "<th style='padding:4px 8px;text-align:left'>SLURM state</th>"
        "<th style='padding:4px 8px;text-align:left'>Elapsed</th>"
        "<th style='padding:4px 8px;text-align:left'>Exit</th>"
        "<th style='padding:4px 8px;text-align:left'>Method/Basis</th>"
        "<th style='padding:4px 8px;text-align:left'>Type</th>"
        "<th style='padding:4px 8px;text-align:left'>Submitted</th>"
        "<th style='padding:4px 8px;text-align:left'>Error</th>"
        "<th style='padding:4px 8px;text-align:left'>Request ID</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )

    options = [_format_slurm_job_option(record) for record in records]
    select.options = options
    select.value = options[0][1]
    select.disabled = False


def _update_slurm_jobs_tab_title(app: Any) -> None:
    root_tab = getattr(app, "root_tab", None)
    tab_index = getattr(app, "_slurm_jobs_tab_index", None)
    if root_tab is None or tab_index is None:
        return
    active = active_slurm_job_count(app)
    title = "Cluster Jobs" if active == 0 else f"Cluster Jobs ({active})"
    try:
        root_tab.set_title(tab_index, title)
    except Exception:
        pass


def on_slurm_jobs_refresh_clicked(app: Any, _btn: Any = None) -> None:
    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)


def on_slurm_jobs_view_clicked(app: Any, _btn: Any = None) -> None:
    select = getattr(app, "_slurm_jobs_select", None)
    request_id = select.value if select is not None else None
    if not request_id:
        return
    attach_slurm_job(app, request_id)
    go_to = getattr(app, "_go_to_calculate_tab", None)
    if callable(go_to):
        go_to()


def on_slurm_jobs_cancel_clicked(app: Any, _btn: Any = None) -> None:
    select = getattr(app, "_slurm_jobs_select", None)
    request_id = select.value if select is not None else None
    if not request_id:
        return
    record = ensure_job_registry(app).load(request_id)
    if record is None:
        return
    if record.status.lower() not in _ACTIVE_SLURM_STATUSES:
        status = getattr(app, "_slurm_jobs_status_html", None)
        if status is not None:
            status.value = (
                f'<span style="color:{_theme.css.TEXT_SUBTLE};font-size:12px">'
                f"Job {request_id} is not active (status: {record.status}).</span>"
            )
        return
    ok = cancel_slurm_job(app, request_id)
    status = getattr(app, "_slurm_jobs_status_html", None)
    if status is not None:
        message = slurm_cancel_user_message(app, request_id)
        color = _theme.css.TEXT_STRONG if ok else _theme.css.ACCENT_ERROR
        status.value = (
            f'<span style="color:{color};font-size:12px">{html.escape(message)}</span>'
        )


def on_slurm_jobs_remove_clicked(app: Any, _btn: Any = None) -> None:
    select = getattr(app, "_slurm_jobs_select", None)
    request_id = select.value if select is not None else None
    if not request_id:
        return
    removed = remove_slurm_job_record(app, request_id)
    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)
    status = getattr(app, "_slurm_jobs_status_html", None)
    if status is not None:
        message = slurm_remove_user_message(app, request_id, removed=removed)
        color = _theme.css.TEXT_STRONG if removed else _theme.css.ACCENT_ERROR
        status.value = (
            f'<span style="color:{color};font-size:12px">{html.escape(message)}</span>'
        )


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

    try:
        from quantui.app_runflow import refresh_results_browser
        from quantui.backends.slurm_ingest import (
            completion_summary_html,
            ingest_staging_success,
        )

        saved_dir = ingest_staging_success(record, log_text)
        app._job_registry.update_status(
            record.request_id, "success", result_dir=str(saved_dir)
        )
        app._last_result_dir = saved_dir
        refresh_results_browser(app)
        calc_type = payload.get("calc_type", record.calc_type)
        app.run_status.value = (
            f"SLURM {calc_type.replace('_', ' ')} complete — saved to {saved_dir.name}."
        )
        _append_run_stdout(app, f"\n✅ Result saved to {saved_dir}\n")
        app.result_output.append_display_data(
            HTML(completion_summary_html(saved_dir, payload))
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
    refresh_slurm_jobs_tab(app)
    _update_slurm_jobs_tab_title(app)
    stop = getattr(app, "_slurm_monitor_stop", None)
    if stop is not None:
        stop.clear()


def _show_slurm_banner(app: Any, message: str, *, request_id: str) -> None:
    banner = getattr(app, "_slurm_job_banner", None)
    if banner is None:
        return
    banner.value = (
        f'<div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;'
        f'padding:10px 12px;margin:6px 0;font-size:13px;color:{_theme.css.TEXT_STRONG}">'
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
