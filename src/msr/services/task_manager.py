from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from threading import RLock
from typing import Any, Callable

from msr.core.config import Settings
from msr.core.errors import QueueFullError, TranscriptionError


TaskProgressCallback = Callable[[str, str | None], None]
TaskRunner = Callable[[TaskProgressCallback], dict[str, Any]]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    status: str
    stage: str
    submitted_at: str
    started_at: str | None
    finished_at: str | None
    queue_wait_ms: int | None
    run_ms: int | None
    filename: str
    audio_duration: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskJob:
    record: TaskRecord
    runner: TaskRunner
    future: asyncio.Future
    submitted_monotonic: float = field(default_factory=time.perf_counter)
    started_monotonic: float | None = None


class TaskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._pending: deque[TaskJob] = deque()
        self._active: dict[str, TaskJob] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._recent: list[TaskRecord] = self._load_recent_tasks()

    async def start(self) -> None:
        self.settings.runtime.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.runtime.task_results_dir.mkdir(parents=True, exist_ok=True)
        self._loop = asyncio.get_running_loop()
        with self._lock:
            self._trim_recent_locked()
            self._persist_recent_tasks_locked()
            self._prune_result_files_locked()

    async def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()

        for job in pending:
            if not job.future.done():
                job.future.set_exception(TranscriptionError("Task manager is shutting down."))

    async def submit(self, task_id: str, filename: str, runner: TaskRunner) -> dict[str, Any]:
        job = await self._submit_job(task_id=task_id, filename=filename, runner=runner)
        return await job.future

    async def submit_async(self, task_id: str, filename: str, runner: TaskRunner) -> dict[str, Any]:
        job = await self._submit_job(task_id=task_id, filename=filename, runner=runner)
        detail = self.task_detail(task_id)
        return detail or self._record_detail(job.record, result_available=False)

    async def _submit_job(self, task_id: str, filename: str, runner: TaskRunner) -> TaskJob:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop

        should_drain = False
        with self._lock:
            if self._closed:
                raise TranscriptionError("Task manager is not accepting new jobs.")

            record = TaskRecord(
                task_id=task_id,
                status="queued",
                stage="queued",
                submitted_at=_utc_now(),
                started_at=None,
                finished_at=None,
                queue_wait_ms=None,
                run_ms=None,
                filename=filename,
            )
            job = TaskJob(record=record, runner=runner, future=loop.create_future())

            if self._pending:
                if len(self._pending) >= self.settings.runtime.max_queued_tasks:
                    raise QueueFullError(self._queue_full_detail_locked())
                self._pending.append(job)
                should_drain = len(self._active) < self.settings.runtime.max_parallel_tasks
            elif len(self._active) < self.settings.runtime.max_parallel_tasks:
                self._start_locked(job, loop)
            else:
                if len(self._pending) >= self.settings.runtime.max_queued_tasks:
                    raise QueueFullError(self._queue_full_detail_locked())
                self._pending.append(job)

        if should_drain:
            self._drain_queue()
        logger.info("Task %s accepted for %s with status=%s", task_id, filename, job.record.status)
        return job

    def limits_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_parallel_tasks": self.settings.runtime.max_parallel_tasks,
                "max_queued_tasks": self.settings.runtime.max_queued_tasks,
                "recent_task_limit": self.settings.runtime.recent_task_limit,
            }

    def task_snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = [job.record.to_dict() for job in self._active.values()]
            queued = [job.record.to_dict() for job in self._pending]
            recent = [record.to_dict() for record in self._recent]
            return {
                "counts": {
                    "active": len(active),
                    "queued": len(queued),
                    "recent": len(recent),
                },
                "active": active,
                "queued": queued,
                "recent": recent,
            }

    def task_detail(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._find_record_locked(task_id)
            if not record:
                return None
            return self._record_detail(record, result_available=self._result_path(task_id).exists())

    def task_result(self, task_id: str) -> tuple[TaskRecord | None, dict[str, Any] | None]:
        with self._lock:
            record = self._find_record_locked(task_id)
            if not record:
                return None, None
            result_path = self._result_path(task_id)

        if not result_path.exists():
            return record, None

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return record, None
        if not isinstance(payload, dict):
            return record, None
        return record, payload

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._active) + len(self._pending)

    def update_limits(
        self,
        *,
        max_parallel_tasks: int | None = None,
        max_queued_tasks: int | None = None,
        recent_task_limit: int | None = None,
    ) -> dict[str, int]:
        with self._lock:
            if max_parallel_tasks is not None:
                self.settings.runtime.max_parallel_tasks = max_parallel_tasks
            if max_queued_tasks is not None:
                self.settings.runtime.max_queued_tasks = max_queued_tasks
            if recent_task_limit is not None:
                self.settings.runtime.recent_task_limit = recent_task_limit
                self._trim_recent_locked()
                self._persist_recent_tasks_locked()
                self._prune_result_files_locked()
            self._persist_runtime_overrides_locked()

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._drain_queue)
        return self.limits_snapshot()

    def _start_locked(self, job: TaskJob, loop: asyncio.AbstractEventLoop) -> None:
        job.started_monotonic = time.perf_counter()
        job.record.status = "running"
        job.record.stage = "starting"
        job.record.started_at = _utc_now()
        job.record.queue_wait_ms = int((job.started_monotonic - job.submitted_monotonic) * 1000)
        self._active[job.record.task_id] = job
        task = loop.create_task(self._run_job(job))
        self._background_tasks[job.record.task_id] = task

    async def _run_job(self, job: TaskJob) -> None:
        try:
            result = await asyncio.to_thread(job.runner, self._build_progress_callback(job.record.task_id))
        except Exception as exc:
            self._mark_failed(job, exc)
            self._delete_result_file(job.record.task_id)
            logger.exception(
                "Task %s failed at stage=%s filename=%s",
                job.record.task_id,
                job.record.stage,
                job.record.filename,
            )
            if not job.future.done():
                job.future.set_exception(exc)
        else:
            self._mark_completed(job, result)
            self._persist_result(job.record.task_id, result)
            if not job.future.done():
                job.future.set_result(result)
        finally:
            with self._lock:
                self._background_tasks.pop(job.record.task_id, None)
                self._active.pop(job.record.task_id, None)
                self._recent.insert(0, job.record)
                self._trim_recent_locked()
                self._persist_recent_tasks_locked()
                self._prune_result_files_locked()

            self._drain_queue()

    def _mark_completed(self, job: TaskJob, result: dict[str, Any]) -> None:
        with self._lock:
            job.record.status = "completed"
            job.record.stage = "completed"
            job.record.finished_at = _utc_now()
            if job.started_monotonic is not None:
                job.record.run_ms = int((time.perf_counter() - job.started_monotonic) * 1000)
            job.record.audio_duration = result.get("audio_duration")
            job.record.error = None
            logger.info(
                "Task %s completed queue_wait_ms=%s run_ms=%s",
                job.record.task_id,
                job.record.queue_wait_ms,
                job.record.run_ms,
            )

    def _mark_failed(self, job: TaskJob, exc: Exception) -> None:
        with self._lock:
            job.record.status = "failed"
            if job.record.stage == "queued":
                job.record.stage = "failed"
            job.record.finished_at = _utc_now()
            if job.started_monotonic is not None:
                job.record.run_ms = int((time.perf_counter() - job.started_monotonic) * 1000)
            job.record.error = str(exc)

    def _build_progress_callback(self, task_id: str) -> TaskProgressCallback:
        def progress(stage: str, audio_duration: str | None = None) -> None:
            with self._lock:
                job = self._active.get(task_id)
                if not job:
                    return
                job.record.stage = stage
                if audio_duration is not None:
                    job.record.audio_duration = audio_duration

        return progress

    def _drain_queue(self) -> None:
        loop = self._loop
        if loop is None:
            return

        with self._lock:
            while self._pending and len(self._active) < self.settings.runtime.max_parallel_tasks:
                next_job = self._pending.popleft()
                self._start_locked(next_job, loop)

    def _trim_recent_locked(self) -> None:
        limit = self.settings.runtime.recent_task_limit
        if len(self._recent) > limit:
            del self._recent[limit:]

    def _record_detail(self, record: TaskRecord, *, result_available: bool) -> dict[str, Any]:
        payload = record.to_dict()
        payload["result_available"] = result_available
        return payload

    def _find_record_locked(self, task_id: str) -> TaskRecord | None:
        active = self._active.get(task_id)
        if active:
            return active.record
        for job in self._pending:
            if job.record.task_id == task_id:
                return job.record
        for record in self._recent:
            if record.task_id == task_id:
                return record
        return None

    def _persist_runtime_overrides_locked(self) -> None:
        path = self.settings.runtime.override_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "[runtime]\n"
            f"max_parallel_tasks = {self.settings.runtime.max_parallel_tasks}\n"
            f"max_queued_tasks = {self.settings.runtime.max_queued_tasks}\n"
            f"recent_task_limit = {self.settings.runtime.recent_task_limit}\n"
        )
        path.write_text(content, encoding="utf-8")

    def _persist_recent_tasks_locked(self) -> None:
        path = self.settings.runtime.recent_tasks_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in self._recent]
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return

    def _persist_result(self, task_id: str, result: dict[str, Any]) -> None:
        path = self._result_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return

    def _delete_result_file(self, task_id: str) -> None:
        path = self._result_path(task_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return

    def _load_recent_tasks(self) -> list[TaskRecord]:
        path = self.settings.runtime.recent_tasks_path
        if not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(payload, list):
            return []

        items = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            try:
                items.append(
                    TaskRecord(
                        task_id=str(raw["task_id"]),
                        status=str(raw["status"]),
                        stage=str(raw["stage"]),
                        submitted_at=str(raw["submitted_at"]),
                        started_at=_optional_str(raw.get("started_at")),
                        finished_at=_optional_str(raw.get("finished_at")),
                        queue_wait_ms=_optional_int(raw.get("queue_wait_ms")),
                        run_ms=_optional_int(raw.get("run_ms")),
                        filename=str(raw.get("filename", "")),
                        audio_duration=_optional_str(raw.get("audio_duration")),
                        error=_optional_str(raw.get("error")),
                    )
                )
            except KeyError:
                continue
        return items[: self.settings.runtime.recent_task_limit]

    def _prune_result_files_locked(self) -> None:
        results_dir = self.settings.runtime.task_results_dir
        if not results_dir.exists():
            return
        keep_ids = {record.task_id for record in self._recent if record.status == "completed"}
        for result_path in results_dir.glob("*.json"):
            if result_path.stem in keep_ids:
                continue
            try:
                result_path.unlink()
            except OSError:
                continue

    def _result_path(self, task_id: str) -> Path:
        return self.settings.runtime.task_result_path(task_id)

    def _queue_full_detail_locked(self) -> dict[str, Any]:
        return {
            "code": "queue_full",
            "message": "Task queue is full. Retry later or raise runtime limits.",
            "max_parallel_tasks": self.settings.runtime.max_parallel_tasks,
            "max_queued_tasks": self.settings.runtime.max_queued_tasks,
            "active_tasks": len(self._active),
            "queued_tasks": len(self._pending),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
