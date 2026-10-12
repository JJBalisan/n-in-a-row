"""Command line front end.

python3 -m tictactoe play --size 5 --k 4 --opponent negamax:5
python3 -m tictactoe match --a mynet:agent --b greedy --games 400
python3 -m tictactoe selfplay --games 20000 --out data.jsonl
python3 -m tictactoe solve --size 3
python3 -m tictactoe bench --size 4 --k 4
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from tictactoe import agents, arena, game, search


def _shape_args(p: argparse.ArgumentParser) -> None:
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


def _shape(args) -> tuple:
    rows = args.rows if args.rows is not None else args.size
    cols = (
        args.cols
        if args.cols is not None
        else (args.rows if args.rows is not None else args.size)
    )
    return rows, cols, args.k


def _color(stream=sys.stdout) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


# ---------------------------------------------------------------- play


def cmd_play(args) -> int:
    rows, cols, k = _shape(args)
    rng = random.Random(args.seed)
    board = game.Board(rows, cols, k)
    engine = agents.load_agent(args.opponent, rng)
    human = agents.HumanAgent()
    human_side = game.X if args.side.upper() == "X" else game.O
    color = _color()

    print(
        "\n%dx%d, %d in a row. You are %s against %s."
        % (
            board.rows,
            board.cols,
            board.k,
            game.SYMBOL[human_side],
            engine.name,
        )
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


def cmd_match(args) -> int:
    rows, cols, k = _shape(args)
    rng = random.Random(args.seed)
    a = agents.load_agent(args.a, random.Random(rng.getrandbits(32)))
    b = agents.load_agent(args.b, random.Random(rng.getrandbits(32)))
    records: list[arena.GameRecord] | None = [] if args.out else None

    started = time.perf_counter()

    def progress(done, total, res):
        if args.quiet or not _color(sys.stderr):
            return
        sys.stderr.write("\r  %d/%d  score %.3f " % (done, total, res.score))
        sys.stderr.flush()

    res = arena.play_match(
        a,
        b,
        args.games,
        rows,
        cols,
        k,
        alternate=not args.no_alternate,
        records=records,
        progress=progress,
    )
    if not args.quiet and _color(sys.stderr):
        sys.stderr.write("\r" + " " * 40 + "\r")

    print(res.summary(a.name, b.name))
    print(
        f"  {time.perf_counter() - started:.2f}s ({args.games / max(1e-9, time.perf_counter() - started):.1f} games/s)"
    )
    if records is not None:
        _write_records(records, args.out)
        print("  wrote %d games to %s" % (len(records), args.out))
    return 0


# ------------------------------------------------------------ selfplay


def cmd_selfplay(args) -> int:
    rows, cols, k = _shape(args)
    started = time.perf_counter()
    counts = {game.X: 0, game.O: 0, game.DRAW: 0}
    n = 0
    handle = open(args.out, "w") if args.out else None
    try:
        for rec in arena.selfplay(
            args.agent, args.games, rows, cols, k, seed=args.seed
        ):
            counts[rec.winner] = counts.get(rec.winner, 0) + 1
            n += 1
            if handle is not None:
                handle.write(rec.to_json() + "\n")
            if not args.quiet and n % 500 == 0 and _color(sys.stderr):
                sys.stderr.write("\r  %d/%d games " % (n, args.games))
                sys.stderr.flush()
    finally:
        if handle is not None:
            handle.close()
    if not args.quiet and _color(sys.stderr):
        sys.stderr.write("\r" + " " * 30 + "\r")
    elapsed = time.perf_counter() - started
    print(
        "%d games of %s in %.2fs (%.0f games/s)"
        % (n, args.agent, elapsed, n / max(1e-9, elapsed))
    )
    print(
        "  X %d   O %d   drawn %d"
        % (
            counts.get(game.X, 0),
            counts.get(game.O, 0),
            counts.get(game.DRAW, 0),
        )
    )
    if args.out:
        print(
            f"  wrote {args.out} -- one JSON game per line; tictactoe.replay() turns each into"
            " (board, move, value) training triples"
        )
    return 0


def _write_records(records, path: str) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


# --------------------------------------------------------------- solve


def cmd_solve(args) -> int:
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
    print(
        "  best %s   %d nodes   %.2fs"
        % (
            board.label(result.move) if result.move is not None else "-",
            engine.nodes,
            time.perf_counter() - started,
        )
    )
    print()
    width = max(len(board.label(m)) for m in per_move) if per_move else 4
    for m in sorted(per_move):
        r = per_move[m]
        print(f"  {board.label(m).rjust(width)}  {r.outcome}")
    return 0


# --------------------------------------------------------------- bench


def cmd_bench(args) -> int:
    rows, cols, k = _shape(args)
    print("board %dx%d, %d in a row" % (rows, cols, k))

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
        "  engine:   %d random games, %d moves in %.2fs -> %.0f moves/s"
        % (args.games, plies, dt, plies / max(1e-9, dt))
    )

    engine = search.Negamax(max_depth=args.depth)
    b = game.Board(rows, cols, k)
    t = time.perf_counter()
    r = engine.search(b)
    dt = time.perf_counter() - t
    print(
        "  search:   depth %d from the empty board, %d nodes in %.2fs -> %.0f nodes/s"
        % (r.depth, r.nodes, dt, r.nodes / max(1e-9, dt))
    )
    print(
        "            {}{}".format(
            r.outcome, "" if r.exact else " (horizon-limited)"
        )
    )

    try:
        # NumPy is optional: without it the batchenv line reports a skip.
        import numpy as np  # noqa: PLC0415

        from tictactoe import env as rl_env  # noqa: PLC0415

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
            "  batchenv: %d envs x %d steps = %d moves in %.2fs -> %.0f moves/s"
            % (args.batch, args.steps, moves, dt, moves / max(1e-9, dt))
        )
    except ImportError:
        print("  batchenv: skipped (numpy not installed)")
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tictactoe",
        description="Generalised tic-tac-toe: play it, search it, or train against it.",
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
