"""Command line front end.

Example:
    python3 -m n_in_a_row play --size 5 --k 4 --opponent negamax:5
    python3 -m n_in_a_row match --a mynet:agent --b greedy --games 400
    python3 -m n_in_a_row selfplay --games 20000 --out data.jsonl
    python3 -m n_in_a_row solve --size 3
    python3 -m n_in_a_row bench --size 4 --k 4
"""

from __future__ import annotations

import argparse
import contextlib
import random
import sys
import time
from collections.abc import Iterable
from typing import TextIO

from n_in_a_row import agents, arena, game, search


def _shape_args(p: argparse.ArgumentParser) -> None:
    """Adds --size, --rows, --cols, -k and --seed to a subcommand."""
    p.add_argument(
        "--size",
        type=int,
        default=3,
        metavar="N",
        help="board is N x N unless --rows/--cols are given (default 3)",
    )
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument(
        "-k",
        "--k",
        type=int,
        default=None,
        metavar="K",
        help="stones in a row needed to win (default: the shorter side)",
    )
    p.add_argument("--seed", type=int, default=None)


def _shape(args: argparse.Namespace) -> tuple:
    """Resolves (rows, cols, k): --rows/--cols beat --size; k may be None."""
    rows = args.rows if args.rows is not None else args.size
    cols = (
        args.cols
        if args.cols is not None
        else (args.rows if args.rows is not None else args.size)
    )
    return rows, cols, args.k


def _color(stream: TextIO = sys.stdout) -> bool:
    """Whether to use ANSI colour and progress output on ``stream``."""
    return hasattr(stream, "isatty") and stream.isatty()


# ---------------------------------------------------------------- play


def cmd_play(args: argparse.Namespace) -> int:
    """Plays a game against an agent in the terminal.

    Args:
        args: Parsed ``play`` arguments.

    Returns:
        0 when the game ends or the player quits.
    """
    rows, cols, k = _shape(args)
    rng = random.Random(args.seed)
    board = game.Board(rows, cols, k)
    engine = agents.load_agent(args.opponent, rng)
    human = agents.HumanAgent()
    human_side = game.X if args.side.upper() == "X" else game.O
    color = _color()

    print(
        f"\n{board.rows}x{board.cols}, {board.k} in a row. "
        f"You are {game.SYMBOL[human_side]} against {engine.name}."
    )
    print("Enter a cell like b2. 'undo' takes back, 'quit' leaves.\n")

    while not board.is_terminal:
        print(board.render(color=color))
        print()
        if board.to_move == human_side:
            try:
                move = human.select(board)
            except agents.Undo:
                if board.ply >= 2:
                    board.pop()
                    board.pop()
                    print("\n(took back a move)\n")
                else:
                    print("\n(nothing to take back)\n")
                continue
            except SystemExit as exc:
                print(exc)
                return 0
        else:
            t = time.perf_counter()
            move = engine.select(board)
            why = engine.explain(board)
            print(
                f"{engine.name} plays {board.label(move)}"
                + (f"   [{why}]" if why and args.verbose else "")
                + (
                    "   (%.2fs)" % (time.perf_counter() - t)
                    if args.verbose
                    else ""
                )
            )
            print()
        board.push(move)

    print(board.render(color=color))
    print()
    if board.winner == game.DRAW:
        print("Drawn.")
    elif board.winner == human_side:
        print(
            "You win on {}.".format(
                "-".join(board.label(i) for i in board.win_line)
            )
        )
    else:
        print(
            "{} wins on {}.".format(
                engine.name, "-".join(board.label(i) for i in board.win_line)
            )
        )
    return 0


# --------------------------------------------------------------- match


def cmd_match(args: argparse.Namespace) -> int:
    """Pits two agents against each other and prints the totals.

    Args:
        args: Parsed ``match`` arguments.

    Returns:
        0.
    """
    rows, cols, k = _shape(args)
    rng = random.Random(args.seed)
    a = agents.load_agent(args.a, random.Random(rng.getrandbits(32)))
    b = agents.load_agent(args.b, random.Random(rng.getrandbits(32)))
    records: list[arena.GameRecord] | None = [] if args.out else None

    started = time.perf_counter()

    def progress(done: int, total: int, res: arena.MatchResult) -> None:
        if args.quiet or not _color(sys.stderr):
            return
        sys.stderr.write(f"\r  {done}/{total}  score {res.score:.3f} ")
        sys.stderr.flush()

    res = arena.play_match(
        a,
        b,
        args.games,
        rows=rows,
        cols=cols,
        k=k,
        alternate=not args.no_alternate,
        records=records,
        progress=progress,
    )
    if not args.quiet and _color(sys.stderr):
        sys.stderr.write("\r" + " " * 40 + "\r")

    print(res.summary(a.name, b.name))
    print(
        f"  {time.perf_counter() - started:.2f}s "
        f"({args.games / max(1e-9, time.perf_counter() - started):.1f}"
        " games/s)"
    )
    if records is not None:
        _write_records(records, args.out)
        print(f"  wrote {len(records)} games to {args.out}")
    return 0


# ------------------------------------------------------------ selfplay


def cmd_selfplay(args: argparse.Namespace) -> int:
    """Generates self-play games, optionally as JSONL training data.

    Args:
        args: Parsed ``selfplay`` arguments.

    Returns:
        0.
    """
    rows, cols, k = _shape(args)
    started = time.perf_counter()
    counts = {game.X: 0, game.O: 0, game.DRAW: 0}
    n = 0
    with (
        open(args.out, "w") if args.out else contextlib.nullcontext()
    ) as handle:
        for rec in arena.selfplay(
            args.agent, args.games, rows, cols, k, seed=args.seed
        ):
            counts[rec.winner] = counts.get(rec.winner, 0) + 1
            n += 1
            if handle is not None:
                handle.write(rec.to_json() + "\n")
            if not args.quiet and n % 500 == 0 and _color(sys.stderr):
                sys.stderr.write(f"\r  {n}/{args.games} games ")
                sys.stderr.flush()
    if not args.quiet and _color(sys.stderr):
        sys.stderr.write("\r" + " " * 30 + "\r")
    elapsed = time.perf_counter() - started
    print(
        f"{n} games of {args.agent} in {elapsed:.2f}s "
        f"({n / max(1e-9, elapsed):.0f} games/s)"
    )
    print(
        f"  X {counts.get(game.X, 0)}   O {counts.get(game.O, 0)}   "
        f"drawn {counts.get(game.DRAW, 0)}"
    )
    if args.out:
        print(
            f"  wrote {args.out} -- one JSON game per line; "
            "n_in_a_row.replay() turns each into"
            " (board, move, value) training triples"
        )
    return 0


def _write_records(records: Iterable[arena.GameRecord], path: str) -> None:
    """Writes one JSON game per line to ``path``, replacing the file."""
    with open(path, "w") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


# --------------------------------------------------------------- solve


def cmd_solve(args: argparse.Namespace) -> int:
    """Evaluates a position and every legal reply.

    Args:
        args: Parsed ``solve`` arguments.

    Returns:
        0.
    """
    rows, cols, k = _shape(args)
    board = game.Board(rows, cols, k)
    for token in args.moves:
        board.push(board.parse_move(token))
    print(board.render(color=_color()))
    print()
    engine = search.Negamax(max_depth=args.depth, time_limit=args.time)
    started = time.perf_counter()
    per_move = engine.evaluate_moves(board)
    result = engine.search(board)
    print(
        "{} to move: {}{}".format(
            game.SYMBOL[board.to_move],
            result.outcome,
            "" if result.exact else "  (horizon-limited estimate)",
        )
    )
    best = board.label(result.move) if result.move is not None else "-"
    print(
        f"  best {best}   {engine.nodes} nodes   "
        f"{time.perf_counter() - started:.2f}s"
    )
    print()
    width = max(len(board.label(m)) for m in per_move) if per_move else 4
    for m in sorted(per_move):
        r = per_move[m]
        print(f"  {board.label(m).rjust(width)}  {r.outcome}")
    return 0


# --------------------------------------------------------------- bench


def cmd_bench(args: argparse.Namespace) -> int:
    """Measures engine, search and (with NumPy) batch-env throughput.

    Args:
        args: Parsed ``bench`` arguments.

    Returns:
        0.
    """
    rows, cols, k = _shape(args)
    print(f"board {rows}x{cols}, {k} in a row")

    t = time.perf_counter()
    plies = 0
    rng = random.Random(args.seed or 0)
    for _ in range(args.games):
        b = game.Board(rows, cols, k)
        while not b.is_terminal:
            b.push(rng.choice(b.legal_moves()))
            plies += 1
    dt = time.perf_counter() - t
    print(
        f"  engine:   {args.games} random games, {plies} moves in "
        f"{dt:.2f}s -> {plies / max(1e-9, dt):.0f} moves/s"
    )

    engine = search.Negamax(max_depth=args.depth)
    b = game.Board(rows, cols, k)
    t = time.perf_counter()
    r = engine.search(b)
    dt = time.perf_counter() - t
    print(
        f"  search:   depth {r.depth} from the empty board, {r.nodes} nodes "
        f"in {dt:.2f}s -> {r.nodes / max(1e-9, dt):.0f} nodes/s"
    )
    print(
        "            {}{}".format(
            r.outcome, "" if r.exact else " (horizon-limited)"
        )
    )

    try:
        # NumPy is optional: without it the batchenv line reports a skip.
        import numpy as np  # noqa: PLC0415

        from n_in_a_row import env as rl_env  # noqa: PLC0415

        env = rl_env.BatchEnv(
            args.batch, rows, cols, k, opponent_spec=None, seed=args.seed
        )
        env.reset()
        t = time.perf_counter()
        moves = 0
        for _ in range(args.steps):
            masks = env.legal_masks()
            actions = [int(np.flatnonzero(m)[0]) for m in masks]
            env.step(actions)
            moves += env.n
        dt = time.perf_counter() - t
        print(
            f"  batchenv: {args.batch} envs x {args.steps} steps = {moves} "
            f"moves in {dt:.2f}s -> {moves / max(1e-9, dt):.0f} moves/s"
        )
    except ImportError:
        print("  batchenv: skipped (numpy not installed)")
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """Builds the parser; each subcommand stores its handler as ``func``.

    Returns:
        The top-level parser.
    """
    p = argparse.ArgumentParser(
        prog="n-in-a-row",
        description=(
            "Generalised n-in-a-row: play it, search it, or train against it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""agent specs:
  random                uniform over legal moves
  greedy                takes a win, blocks a loss, else plays centrally
  perfect               alpha-beta, exact on small boards
  negamax:6             alpha-beta to depth 6
  time:0.5              alpha-beta with a per-move time budget in seconds
  noisy:0.2:perfect     any agent, played at random 20%% of the time
  mymodule:factory      your own -- an Agent, an Agent subclass, or a factory
""",
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("play", help="play against an agent in the terminal")
    _shape_args(q)
    q.add_argument("--opponent", default="perfect")
    q.add_argument("--side", default="X", choices=["X", "O", "x", "o"])
    q.add_argument(
        "-v", "--verbose", action="store_true", help="show what the engine saw"
    )
    q.set_defaults(func=cmd_play)

    q = sub.add_parser("match", help="pit two agents against each other")
    _shape_args(q)
    q.add_argument("--a", required=True, help="agent spec for player A")
    q.add_argument("--b", required=True, help="agent spec for player B")
    q.add_argument("--games", type=int, default=100)
    q.add_argument(
        "--no-alternate",
        action="store_true",
        help="do not swap colours between games",
    )
    q.add_argument(
        "--out",
        default=None,
        metavar="FILE.jsonl",
        help="also record the games",
    )
    q.add_argument("--quiet", action="store_true")
    q.set_defaults(func=cmd_match)

    q = sub.add_parser("selfplay", help="generate games as training data")
    _shape_args(q)
    q.add_argument(
        "--agent",
        default="noisy:0.25:perfect",
        help="agent playing both sides (default: noisy:0.25:perfect)",
    )
    q.add_argument("--games", type=int, default=1000)
    q.add_argument("--out", default=None, metavar="FILE.jsonl")
    q.add_argument("--quiet", action="store_true")
    q.set_defaults(func=cmd_selfplay)

    q = sub.add_parser(
        "solve", help="evaluate a position and every legal reply"
    )
    _shape_args(q)
    q.add_argument(
        "moves", nargs="*", help="moves already played, e.g. b2 a1 c3"
    )
    q.add_argument("--depth", type=int, default=None)
    q.add_argument("--time", type=float, default=None, metavar="SECONDS")
    q.set_defaults(func=cmd_solve)

    q = sub.add_parser(
        "bench", help="measure engine, search and env throughput"
    )
    _shape_args(q)
    q.add_argument("--games", type=int, default=2000)
    q.add_argument("--depth", type=int, default=6)
    q.add_argument("--batch", type=int, default=256)
    q.add_argument("--steps", type=int, default=100)
    q.set_defaults(func=cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: Arguments without the program name; sys.argv[1:] if None.

    Returns:
        The exit code: 0 on success, 2 for an invalid move or bad
        arguments (argparse raises SystemExit itself), 130 on Ctrl-C.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except game.InvalidMove as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
