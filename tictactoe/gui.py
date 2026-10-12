"""Tkinter front end, built on the same engine the CLI and the RL env use.

    python3 -m tictactoe.gui --size 5 --k 4 --opponent negamax:5
    python3 -m tictactoe.gui --opponent mynet:agent

Nothing here knows how an opponent decides anything -- it only calls
``select(board)``, so your network drops in exactly where 'perfect' sits.
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import random
import threading
import time

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tkinter is not available for this interpreter.\n"
        "  macOS/Homebrew:  brew install python-tk\n"
        "  Debian/Ubuntu:   sudo apt install python3-tk\n"
        "The CLI (python3 -m tictactoe) needs no extra packages."
    ) from exc

from tictactoe import agents, game, search

BG = "#12150F"
GRID = "#3A4235"
CELL = "#191D16"
CELL_HOVER = "#212619"
INK = "#E7EBDD"
MUTED = "#8B9382"
X_COLOR = "#7FD1E0"
O_COLOR = "#EC9455"
WIN_COLOR = "#93D6A5"
GOOD = "#7FBE94"
EVEN = "#8B9382"
BAD = "#DE7B66"

PRESETS = [
    "human",
    "random",
    "greedy",
    "perfect",
    "negamax:4",
    "negamax:6",
    "time:1.0",
    "noisy:0.2:perfect",
]


class TicTacToeGUI:
    def __init__(
        self,
        root: tk.Tk,
        rows: int,
        cols: int,
        k: int,
        opponent_spec: str,
        human_side: int,
        show_solver: bool,
        seed=None,
    ):
        self.root = root
        self.rng = random.Random(seed)
        self.board = game.Board(rows, cols, k)
        self.opponent_spec = opponent_spec
        self.human_side = human_side
        self.agent: agents.Agent | None = None
        self.thinking = False
        self.queue: queue.Queue = queue.Queue()
        self.solver_values: dict[int, object] = {}
        self.solver_engine = search.Negamax(max_depth=None, time_limit=2.0)
        self.hover: int | None = None

        root.title("Tic-Tac-Toe")
        root.configure(bg=BG)
        root.minsize(520, 620)

        self._build_controls()
        self._build_board()
        self._build_status()

        self.show_solver = tk.BooleanVar(value=show_solver)
        self.solver_check.configure(variable=self.show_solver)
        self.show_solver.trace_add(
            "write", lambda *_: self.new_game(keep_position=True)
        )

        self.new_game()
        self.root.after(60, self._poll)

    # ---- widgets --------------------------------------------------------

    def _build_controls(self) -> None:
        style = ttk.Style()
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")
        style.configure("T.TFrame", background=BG)
        style.configure(
            "T.TLabel", background=BG, foreground=MUTED, font=("Helvetica", 11)
        )
        style.configure(
            "T.TCheckbutton",
            background=BG,
            foreground=MUTED,
            font=("Helvetica", 11),
        )
        style.map(
            "T.TCheckbutton",
            background=[("active", BG)],
            foreground=[("active", INK)],
        )

        bar = ttk.Frame(self.root, style="T.TFrame", padding=(16, 14, 16, 6))
        bar.pack(fill="x")

        def label(parent, text):
            return ttk.Label(parent, text=text, style="T.TLabel")

        row1 = ttk.Frame(bar, style="T.TFrame")
        row1.pack(fill="x", pady=(0, 8))
        label(row1, "rows").pack(side="left")
        self.rows_var = tk.IntVar(value=self.board.rows)
        tk.Spinbox(
            row1,
            from_=3,
            to=15,
            width=3,
            textvariable=self.rows_var,
            bg=CELL,
            fg=INK,
            buttonbackground=GRID,
            relief="flat",
            highlightthickness=0,
            insertbackground=INK,
        ).pack(side="left", padx=(6, 14))
        label(row1, "cols").pack(side="left")
        self.cols_var = tk.IntVar(value=self.board.cols)
        tk.Spinbox(
            row1,
            from_=3,
            to=15,
            width=3,
            textvariable=self.cols_var,
            bg=CELL,
            fg=INK,
            buttonbackground=GRID,
            relief="flat",
            highlightthickness=0,
            insertbackground=INK,
        ).pack(side="left", padx=(6, 14))
        label(row1, "in a row").pack(side="left")
        self.k_var = tk.IntVar(value=self.board.k)
        tk.Spinbox(
            row1,
            from_=3,
            to=15,
            width=3,
            textvariable=self.k_var,
            bg=CELL,
            fg=INK,
            buttonbackground=GRID,
            relief="flat",
            highlightthickness=0,
            insertbackground=INK,
        ).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(bar, style="T.TFrame")
        row2.pack(fill="x")
        label(row2, "opponent").pack(side="left")
        self.opp_var = tk.StringVar(value=self.opponent_spec)
        combo = ttk.Combobox(
            row2,
            textvariable=self.opp_var,
            width=18,
            values=PRESETS
            + (
                [self.opponent_spec]
                if self.opponent_spec not in PRESETS
                else []
            ),
        )
        combo.pack(side="left", padx=(6, 14))
        label(row2, "you play").pack(side="left")
        self.side_var = tk.StringVar(
            value="X" if self.human_side == game.X else "O"
        )
        ttk.Combobox(
            row2,
            textvariable=self.side_var,
            width=3,
            values=["X", "O"],
            state="readonly",
        ).pack(side="left", padx=(6, 14))

        tk.Button(
            row2,
            text="New game",
            command=self.new_game,
            bg=CELL,
            fg=INK,
            activebackground=GRID,
            activeforeground=INK,
            relief="flat",
            highlightthickness=0,
            padx=12,
            pady=3,
        ).pack(side="right")
        tk.Button(
            row2,
            text="Undo",
            command=self.undo,
            bg=CELL,
            fg=MUTED,
            activebackground=GRID,
            activeforeground=INK,
            relief="flat",
            highlightthickness=0,
            padx=12,
            pady=3,
        ).pack(side="right", padx=(0, 8))

        row3 = ttk.Frame(bar, style="T.TFrame")
        row3.pack(fill="x", pady=(8, 0))
        self.solver_check = ttk.Checkbutton(
            row3, text="show the solver's read", style="T.TCheckbutton"
        )
        self.solver_check.pack(side="left")

    def _build_board(self) -> None:
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=8)
        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", lambda e: self._set_hover(None))

    def _build_status(self) -> None:
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=16, pady=(0, 14))
        self.status = tk.Label(
            bar, text="", bg=BG, fg=INK, anchor="w", font=("Helvetica", 13)
        )
        self.status.pack(side="left")
        self.detail = tk.Label(
            bar, text="", bg=BG, fg=MUTED, anchor="e", font=("Helvetica", 11)
        )
        self.detail.pack(side="right")

    # ---- game flow ------------------------------------------------------

    def new_game(self, keep_position: bool = False) -> None:
        if not keep_position:
            rows = max(3, min(15, int(self.rows_var.get())))
            cols = max(3, min(15, int(self.cols_var.get())))
            k = max(2, min(max(rows, cols), int(self.k_var.get())))
            self.k_var.set(k)
            self.board = game.Board(rows, cols, k)
            self.opponent_spec = self.opp_var.get().strip() or "perfect"
            self.human_side = game.X if self.side_var.get() == "X" else game.O
            try:
                self.agent = (
                    None
                    if self.opponent_spec == "human"
                    else agents.load_agent(self.opponent_spec, self.rng)
                )
            except Exception as exc:
                messagebox.showerror("Cannot load opponent", str(exc))
                self.opponent_spec = "perfect"
                self.opp_var.set("perfect")
                self.agent = agents.load_agent("perfect", self.rng)
        self.solver_values = {}
        self.thinking = False
        self.draw()
        self.after_move()

    def undo(self) -> None:
        if self.thinking:
            return
        steps = 1 if self.agent is None else 2
        for _ in range(steps):
            if self.board.ply:
                self.board.pop()
        self.solver_values = {}
        self.draw()
        self.after_move()

    def human_to_move(self) -> bool:
        if self.board.is_terminal or self.thinking:
            return False
        return self.agent is None or self.board.to_move == self.human_side

    def on_click(self, event) -> None:
        if not self.human_to_move():
            return
        idx = self.cell_at(event.x, event.y)
        if idx is None or self.board.cells[idx] != game.EMPTY:
            return
        self.board.push(idx)
        self.solver_values = {}
        self.draw()
        self.after_move()

    def on_motion(self, event) -> None:
        self._set_hover(
            self.cell_at(event.x, event.y) if self.human_to_move() else None
        )

    def _set_hover(self, idx) -> None:
        if idx is not None and self.board.cells[idx] != game.EMPTY:
            idx = None
        if idx != self.hover:
            self.hover = idx
            self.draw()

    def after_move(self) -> None:
        """Kick off whatever background work the new position calls for."""
        if self.board.is_terminal:
            self.draw()
            return
        if self.agent is not None and self.board.to_move != self.human_side:
            self.thinking = True
            self.draw()
            threading.Thread(target=self._think, daemon=True).start()
        elif self.show_solver.get():
            threading.Thread(target=self._read_position, daemon=True).start()

    def _think(self) -> None:
        snapshot = self.board.copy()
        started = time.perf_counter()
        try:
            move = self.agent.select(snapshot)
            why = self.agent.explain(snapshot)
        except Exception as exc:
            self.queue.put(("error", str(exc)))
            return
        # Keep a slow-looking engine from feeling instant and a fast one from
        # flickering the board out from under the click.
        remaining = 0.22 - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        self.queue.put(("move", move, why))

    def _read_position(self) -> None:
        snapshot = self.board.copy()
        try:
            values = self.solver_engine.evaluate_moves(snapshot)
        except Exception:
            return
        self.queue.put(("solver", snapshot.hash, values))

    def _poll(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "move":
                    _, move, why = item
                    self.thinking = False
                    if (
                        not self.board.is_terminal
                        and self.board.cells[move] == game.EMPTY
                    ):
                        self.board.push(move)
                    self.detail.configure(text=why or "")
                    self.draw()
                    self.after_move()
                elif item[0] == "solver":
                    _, board_hash, values = item
                    if board_hash == self.board.hash:
                        self.solver_values = values
                        self.draw()
                elif item[0] == "error":
                    self.thinking = False
                    messagebox.showerror("Opponent failed", item[1])
                    self.draw()
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    # ---- geometry -------------------------------------------------------

    def metrics(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        cell = max(12, min(w / self.board.cols, h / self.board.rows))
        bw, bh = cell * self.board.cols, cell * self.board.rows
        return (w - bw) / 2.0, (h - bh) / 2.0, cell

    def cell_at(self, x, y):
        ox, oy, cell = self.metrics()
        c = int((x - ox) // cell)
        r = int((y - oy) // cell)
        if 0 <= r < self.board.rows and 0 <= c < self.board.cols:
            return self.board.index(r, c)
        return None

    # ---- painting -------------------------------------------------------

    def draw(self) -> None:
        cv = self.canvas
        cv.delete("all")
        ox, oy, cell = self.metrics()
        b = self.board
        win = set(b.win_line or ())
        pad = cell * 0.19
        lw = max(2, int(cell * 0.055))

        for i in range(b.size):
            r, c = b.coord(i)
            x0, y0 = ox + c * cell, oy + r * cell
            x1, y1 = x0 + cell, y0 + cell
            fill = CELL
            if i in win:
                fill = "#1E2A20"
            elif i == self.hover and b.cells[i] == game.EMPTY:
                fill = CELL_HOVER
            cv.create_rectangle(
                x0 + 1, y0 + 1, x1 - 1, y1 - 1, fill=fill, outline=GRID, width=1
            )

            v = b.cells[i]
            if v == game.X:
                col = WIN_COLOR if i in win else X_COLOR
                cv.create_line(
                    x0 + pad,
                    y0 + pad,
                    x1 - pad,
                    y1 - pad,
                    fill=col,
                    width=lw,
                    capstyle="round",
                )
                cv.create_line(
                    x1 - pad,
                    y0 + pad,
                    x0 + pad,
                    y1 - pad,
                    fill=col,
                    width=lw,
                    capstyle="round",
                )
            elif v == game.O:
                col = WIN_COLOR if i in win else O_COLOR
                cv.create_oval(
                    x0 + pad,
                    y0 + pad,
                    x1 - pad,
                    y1 - pad,
                    outline=col,
                    width=lw,
                )
            elif self.show_solver.get() and i in self.solver_values:
                tag, colour = self._solver_tag(self.solver_values[i])
                cv.create_text(
                    x0 + cell * 0.16,
                    y0 + cell * 0.15,
                    text=tag,
                    fill=colour,
                    anchor="nw",
                    font=("Menlo", max(8, int(cell * 0.13)), "bold"),
                )

        if b.last_move is not None and b.last_move not in win:
            r, c = b.coord(b.last_move)
            cv.create_rectangle(
                ox + c * cell + 3,
                oy + r * cell + 3,
                ox + (c + 1) * cell - 3,
                oy + (r + 1) * cell - 3,
                outline=MUTED,
                width=1,
            )
        self._update_status()

    def _solver_tag(self, result):
        if not result.exact:
            return ("%+.1f" % (result.value / 1000.0), EVEN)
        if abs(result.value) < search.WIN // 2:
            return ("D", EVEN)
        n = (search.WIN - abs(result.value) + 1) // 2
        return ("W%d" % n, GOOD) if result.value > 0 else ("L%d" % n, BAD)

    def _update_status(self) -> None:
        b = self.board
        if b.is_terminal:
            if b.winner == game.DRAW:
                self.status.configure(text="Drawn.", fg=INK)
            else:
                who = (
                    "You"
                    if (self.agent is not None and b.winner == self.human_side)
                    else ("Player %s" % ("X" if b.winner == game.X else "O"))
                    if self.agent is None
                    else self.opponent_spec
                )
                line = "-".join(b.label(i) for i in b.win_line)
                self.status.configure(
                    text=f"{who} won on {line}.", fg=WIN_COLOR
                )
            self.detail.configure(text="%d moves" % b.ply)
            return
        if self.thinking:
            self.status.configure(
                text=f"{self.opponent_spec} is thinking…", fg=MUTED
            )
            return
        if self.agent is None:
            self.status.configure(
                text="Player %s to move."
                % ("X" if b.to_move == game.X else "O"),
                fg=INK,
            )
        else:
            self.status.configure(
                text="Your move (%s)."
                % ("X" if self.human_side == game.X else "O"),
                fg=INK,
            )
        self.detail.configure(text="%dx%d, %d in a row" % (b.rows, b.cols, b.k))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tictactoe.gui", description="Play tic-tac-toe in a window."
    )
    p.add_argument("--size", type=int, default=3, metavar="N")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument(
        "--opponent",
        default="perfect",
        help="agent spec: random, greedy, perfect, negamax:D, time:S, "
        "noisy:P:AGENT, human, or module:attribute",
    )
    p.add_argument("--side", default="X", choices=["X", "O", "x", "o"])
    p.add_argument(
        "--solver", action="store_true", help="start with the solver overlay on"
    )
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    rows = args.rows if args.rows is not None else args.size
    cols = args.cols if args.cols is not None else rows
    k = args.k if args.k is not None else min(rows, cols)

    root = tk.Tk()
    TicTacToeGUI(
        root,
        rows,
        cols,
        k,
        args.opponent,
        game.X if args.side.upper() == "X" else game.O,
        args.solver or (rows * cols <= 9),
        args.seed,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
