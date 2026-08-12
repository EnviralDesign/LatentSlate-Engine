"""Rich-only presentation adapter for acquisition progress events."""

from __future__ import annotations

from contextlib import AbstractContextManager
from threading import Lock
from typing import TYPE_CHECKING, Any, Self

from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TextColumn,
)
from rich.text import Text

from .cli_presentation import human_console

if TYPE_CHECKING:
    from rich.console import Console


class HumanInstallProgress(AbstractContextManager["HumanInstallProgress"]):
    """Thread-safe adapter used only while an interactive install command runs."""

    def __init__(self, console: Console | None = None) -> None:
        self._lock = Lock()
        self._console = console or human_console()
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            DownloadColumn(),
            _KnownSpeedColumn(),
            _KnownEtaColumn(),
            console=self._console,
            transient=False,
        )
        self._live = Live(self._progress, console=self._console, refresh_per_second=10)
        self._overall: TaskID | None = None
        self._resource_task: TaskID | None = None
        self._active_resource_id: str | None = None
        self._completed_resources: set[str] = set()
        self._hf_tasks: dict[int, TaskID] = {}

    def __enter__(self) -> Self:
        self._live.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._live.stop()

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        """Accept presentation-neutral acquisition events."""

        with self._lock:
            if event == "preflight":
                total = int(data["resource_count"])
                self._overall = self._progress.add_task("Preflight", total=total, completed=0)
            elif event == "skipped":
                resource_id = str(data["resource_id"])
                if resource_id not in self._completed_resources:
                    self._completed_resources.add(resource_id)
                    self._advance_overall(f"Skipped · {resource_id}")
            elif event == "resource_start":
                resource_id = str(data["resource_id"])
                self._remove_resource_task()
                self._active_resource_id = resource_id
                label = f"Downloading · {resource_id}"
                self._resource_task = self._progress.add_task(
                    label, total=_known_total(data.get("total_bytes"))
                )
            elif event == "phase":
                resource_id = str(data["resource_id"])
                self._set_active_description(
                    resource_id, f"{data['phase'].capitalize()} · {resource_id}"
                )
            elif event == "download_progress":
                resource_id = str(data["resource_id"])
                task = self._active_resource_task(resource_id)
                if task is not None:
                    self._progress.update(
                        task.id,
                        total=_known_total(data.get("total")),
                        completed=int(data["completed"]),
                    )
            elif event == "complete":
                resource_id = str(data["resource_id"])
                if resource_id in self._completed_resources:
                    return
                task = self._active_resource_task(resource_id)
                if task is not None:
                    if task.total is not None:
                        self._progress.update(task.id, completed=task.total)
                    self._progress.update(task.id, description=f"Complete · {resource_id}")
                    self._clear_hf_tasks()
                    self._active_resource_id = None
                    self._completed_resources.add(resource_id)
                    self._advance_overall(f"Complete · {resource_id}")
            elif event == "failed":
                resource_id = str(data["resource_id"])
                self._set_active_description(resource_id, f"Failed · {resource_id}")
            self._live.refresh()

    @property
    def tqdm_class(self) -> type[_RichTqdm]:
        """Supported ``huggingface_hub`` hook; a fresh wrapper is safe per worker."""

        presenter = self

        class _BoundRichTqdm(_RichTqdm):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(presenter, *args, **kwargs)

        return _BoundRichTqdm

    def _advance_overall(self, description: str) -> None:
        task = self._task_or_none(self._overall)
        if task is not None:
            self._progress.update(task.id, advance=1, description=description)

    def _set_active_description(self, resource_id: str, description: str) -> None:
        task = self._active_resource_task(resource_id)
        if task is not None:
            self._progress.update(task.id, description=description)

    def _active_resource_task(self, resource_id: str) -> Any | None:
        if self._active_resource_id != resource_id:
            return None
        return self._task_or_none(self._resource_task)

    def _task_or_none(self, task_id: TaskID | None) -> Any | None:
        """Look up Rich's opaque task identifier without indexing its task list.

        Rich 14 exposes the ID-keyed mapping internally but no public single-task
        accessor.  ``tasks`` is display-order only, so indexing it is incorrect
        after task removal or reordering.
        """

        if task_id is None:
            return None
        return self._progress._tasks.get(task_id)

    def _tqdm_start(
        self, bar: object, description: str, total: float | None, initial: float
    ) -> TaskID:
        with self._lock:
            task_id = self._progress.add_task(
                description, total=_known_total(total), completed=initial
            )
            self._hf_tasks[id(bar)] = task_id
            return task_id

    def _tqdm_update(self, task_id: TaskID, amount: float) -> None:
        with self._lock:
            task = self._task_or_none(task_id)
            if task is not None:
                self._progress.update(task.id, advance=amount)
            self._live.refresh()

    def _tqdm_reset(self, bar: object, total: float | None) -> None:
        with self._lock:
            task_id = self._hf_tasks.get(id(bar))
            task = self._task_or_none(task_id)
            if task is not None:
                self._progress.reset(task.id, total=_known_total(total))

    def _tqdm_description(self, bar: object, description: str) -> None:
        with self._lock:
            task_id = self._hf_tasks.get(id(bar))
            task = self._task_or_none(task_id)
            if task is not None:
                self._progress.update(task.id, description=description)

    def _tqdm_total(self, bar: object, total: float | None) -> None:
        with self._lock:
            task_id = self._hf_tasks.get(id(bar))
            task = self._task_or_none(task_id)
            if task is not None:
                self._progress.update(task.id, total=_known_total(total))

    def _tqdm_close(self, bar: object) -> None:
        with self._lock:
            task_id = self._hf_tasks.pop(id(bar), None)
            if self._task_or_none(task_id) is not None:
                self._progress.remove_task(task_id)

    def _remove_resource_task(self) -> None:
        if self._task_or_none(self._resource_task) is not None:
            self._progress.remove_task(self._resource_task)
        self._resource_task = None
        self._active_resource_id = None

    def _clear_hf_tasks(self) -> None:
        for task_id in self._hf_tasks.values():
            if self._task_or_none(task_id) is not None:
                self._progress.remove_task(task_id)
        self._hf_tasks.clear()


class _KnownEtaColumn(ProgressColumn):
    """Suppress Rich's unknown ETA placeholder for indeterminate Xet phases."""

    def render(self, task: Any) -> Text:
        if task.total is None or task.speed is None or task.speed <= 0:
            return Text("")
        remaining = max(0, task.total - task.completed) / task.speed
        minutes, seconds = divmod(round(remaining), 60)
        return Text(f"ETA {minutes}:{seconds:02d}", style="muted")


class _KnownSpeedColumn(ProgressColumn):
    """Render a transfer rate only once Rich has one; never show ``?``."""

    def render(self, task: Any) -> Text:
        if task.speed is None or task.speed <= 0:
            return Text("")
        return Text(f"{task.speed / 1_000_000:.1f} MB/s", style="muted")


def _known_total(value: object) -> float | None:
    """Treat zero/unknown upstream totals as indeterminate, never as a 0 B bar."""

    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


class _RichTqdm:
    """Minimal, lock-backed tqdm-compatible adapter for HF worker threads."""

    def __init__(self, presenter: HumanInstallProgress, *args: Any, **kwargs: Any) -> None:
        self._presenter = presenter
        self._total = kwargs.get("total")
        self.n = kwargs.get("initial", 0)
        self.desc = str(kwargs.get("desc") or "Downloading file")
        self.postfix = ""
        self.disable = bool(kwargs.get("disable", False))
        self.leave = bool(kwargs.get("leave", False))
        self._closed = False
        self._task_id = presenter._tqdm_start(self, self.desc, self._total, self.n)
        self.iterable = args[0] if args else kwargs.get("iterable")

    @property
    def total(self) -> float | None:
        return self._total

    @total.setter
    def total(self, value: float | None) -> None:
        self._total = value
        self._presenter._tqdm_total(self, value)

    @property
    def format_dict(self) -> dict[str, Any]:
        return {"n": self.n, "total": self.total, "rate": None}

    def update(self, amount: float = 1) -> None:
        self.n += amount
        self._presenter._tqdm_update(self._task_id, amount)

    def update_transfer(self, amount: float = 1) -> None:
        """Accept Xet's paired transfer callback without double-counting bytes.

        Hugging Face/Xet reports the same chunk through ``update`` (reconstructed
        bytes) and ``update_transfer`` (network bytes).  The compact human view
        intentionally has one byte row, so ``update`` is authoritative.
        """

        return

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._presenter._tqdm_close(self)

    def set_description(self, description: str, *args: Any, **kwargs: Any) -> None:
        self.desc = description
        self._presenter._tqdm_description(self, description)

    def set_description_str(self, description: str | None = None, refresh: bool = True) -> None:
        self.set_description(description or "")

    def set_postfix_str(self, postfix: str = "", refresh: bool = True) -> None:
        self.postfix = postfix

    def set_transfer_postfix_str(self, postfix: str = "", refresh: bool = True) -> None:
        self.set_postfix_str(postfix, refresh)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        values = ordered_dict if isinstance(ordered_dict, dict) else kwargs
        self.postfix = ", ".join(f"{key}={value}" for key, value in values.items())

    def reset(self, total: float | None = None) -> None:
        self.n = 0
        if total is not None:
            self.total = total
        self._presenter._tqdm_reset(self, self.total)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        return None

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        self._presenter._live.refresh()

    @classmethod
    def get_lock(cls) -> Lock:
        return _TQDM_LOCK

    @classmethod
    def set_lock(cls, lock: Lock) -> None:
        global _TQDM_LOCK
        _TQDM_LOCK = lock

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            yield item
            self.update()


_TQDM_LOCK = Lock()
