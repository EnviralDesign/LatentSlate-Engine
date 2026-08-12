"""Small, shared Rich primitives for human-facing CLI output.

Machine-readable commands deliberately do not use this module: their JSON is
written directly by the command dispatcher.  Keeping the presentation boundary
here makes that contract easy to audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_THEME = Theme(
    {
        "heading": "bold bright_cyan",
        "label": "bold cyan",
        "muted": "dim",
        "status.ok": "bold green",
        "status.warn": "bold yellow",
        "status.bad": "bold red",
        "command": "bold bright_white on rgb(40,40,52)",
        "identifier": "bright_cyan",
    }
)


def print_human(renderable: RenderableType) -> None:
    """Print a human view, respecting Rich's normal TTY and ``NO_COLOR`` behavior."""

    Console(theme=_THEME, highlight=False, markup=False).print(renderable)


def render_human(renderable: RenderableType, *, width: int = 100) -> str:
    """Render plain, deterministic text for tests and non-interactive consumers."""

    console = Console(
        file=StringIO(),
        theme=_THEME,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
        markup=False,
        record=True,
    )
    console.print(renderable)
    return console.export_text()


def page(title: str, *sections: RenderableType) -> Group:
    """Group a titled CLI page with consistent vertical breathing room."""

    return Group(Text(title, style="heading"), *sections)


def panel(title: str, body: RenderableType, *, style: str = "cyan") -> Panel:
    """Return a compact section panel that safely wraps at narrow widths."""

    return Panel(body, title=Text(title, style="heading"), border_style=style, expand=True)


def key_values(rows: Iterable[tuple[str, str | Text]]) -> Table:
    """Create a label/value grid that folds long identities and file paths."""

    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="label", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(f"{label}:", value)
    return table


def data_table(*headers: str, ratio: tuple[int | None, ...] | None = None) -> Table:
    """Create a compact table whose data folds instead of being omitted."""

    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_edge=False,
        pad_edge=False,
        expand=True,
    )
    for index, header in enumerate(headers):
        table.add_column(
            header,
            overflow="fold",
            ratio=ratio[index] if ratio else None,
        )
    return table


def status(value: str, kind: str) -> Text:
    """Return a consistently styled status badge without embedding ANSI codes."""

    return Text(value, style=f"status.{kind}")


def identifier(value: str) -> Text:
    """Return a foldable identifier cell."""

    return Text(value, style="identifier")


def next_action(command: str, *, label: str = "Next") -> Panel:
    """Make the actionable follow-up visibly separate from diagnostic detail."""

    return Panel(
        Text(command, style="command"),
        title=Text(label, style="heading"),
        border_style="green",
        expand=True,
    )


def bullet_list(items: Iterable[str], *, style: str = "") -> Text:
    """Render one wrapped bullet per item without requiring a wide terminal."""

    text = Text(style=style)
    for index, item in enumerate(items):
        if index:
            text.append("\n")
        text.append("• ")
        text.append(item)
    return text
