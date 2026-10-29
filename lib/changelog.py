"""Keeps a repo's unreleased CHANGELOG entries in one file per change.

When every PR adds its lines under ``## [Unreleased]`` in CHANGELOG.md, any
two open PRs conflict there, and each merge needs a rebase and a hand
resolution (launchd-jobs, 28 Oct 2026: all ten pairs of five PRs). GitHub's
merge button ignores .gitattributes merge drivers, so the fix is to stop
sharing the lines: each change adds its own file under changes/, and a
release folds them into CHANGELOG.md.

A fragment is changes/<number>-<slug>.md (the issue or PR number, then a
few words), holding Keep a Changelog sections and their bullets only:

    ### Fixed

    - `gh-runners` logs why a scan failed (#119). A bullet may run
      over several lines, each continuation indented.

Sections: Added, Changed, Deprecated, Removed, Fixed, Security, any number
and any order. changes/README.md and changes/.gitkeep are not fragments.

    changelog.py check
        Checks every fragment, and the entries still written under
        [Unreleased] in CHANGELOG.md, against that shape. Once the repo has
        adopted fragments (changes/README.md exists), anything under
        [Unreleased] is itself an error: two PRs that both add lines there
        conflict. Exit 1 on any error, each one named as `file:line: why`
        on stderr.
    changelog.py show
        Prints the sections the next release would add.
    changelog.py new SLUG [--section S]
        Writes changes/<n>-SLUG.md, n being the number the next issue or
        PR will get (the newest one's plus one, from `gh api`, run in the
        repo so gh names it from the remote), holding a `### S` heading
        (Changed by default) and a `- TODO (#n).` bullet that `check`
        refuses until it is filled in. Without gh, or when gh fails, it
        writes changes/SLUG.md, which folds after the numbered ones; rename
        it once the PR has a number. Prints the path it wrote; never
        overwrites. SLUG is lowercase words joined by hyphens.
    changelog.py release X.Y.Z [--date YYYY-MM-DD]
        Folds [Unreleased]'s entries, then the fragments, into a new
        `## [X.Y.Z] - date` section below an empty [Unreleased], in section
        order and, within a section, in fragment order (leading number,
        then name); rewrites the compare links at the bottom; deletes the
        fragments. Refuses, changing nothing, if anything fails to parse
        or nothing is waiting. Then commit, merge, tag vX.Y.Z as before.

Exit 64 on usage. Standard library only, Python 3.9 or later.

Copying it: put the file one directory below the repo root (lib/, ci/,
tools/) beside a changes/ folder at the root; edit REPO_ROOT when it lives
elsewhere. The canonical copy is dev-toolchain's templates/changelog.py.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = 'CHANGELOG.md'
CHANGES = 'changes'
NOT_FRAGMENTS = ('README.md', '.gitkeep')
SECTIONS = ('Added', 'Changed', 'Deprecated', 'Removed', 'Fixed', 'Security')
EX_USAGE = 64
USAGE = (
    'usage: changelog.py check | show | new SLUG [--section S] | '
    'release X.Y.Z [--date YYYY-MM-DD]'
)
DEFAULT_SECTION = 'Changed'
TODO = '- TODO'
# The newest issue or PR, open or closed; gh fills {owner}/{repo} from the
# remote of the directory it runs in.
NEWEST = (
    'gh',
    'api',
    'repos/{owner}/{repo}/issues?state=all&per_page=1',
    '--jq',
    '.[0].number',
)

UNRELEASED = '## [Unreleased]'
RELEASE_HEADING = re.compile(r'^## \[(\d+\.\d+\.\d+)\]', re.MULTILINE)
UNRELEASED_END = re.compile(r'^(## |\[[^\]]+\]: )', re.MULTILINE)
UNRELEASED_LINK = re.compile(
    r'^\[Unreleased\]: (https?://\S+?)/(?:compare|commits|tree)/\S*$',
    re.MULTILINE,
)
VERSION = re.compile(r'^\d+\.\d+\.\d+$')
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
SLUG = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+)*$')

Entries = dict[str, list[str]]


class ChangelogError(Exception):
    """Something the user must fix; the message says what and where."""


def parse(text: str, where: str, offset: int = 0) -> Entries:
    """Parses fragment-shaped text into entries per section.

    Args:
        text: Sections and bullets only, as a fragment holds.
        where: How to name the source in an error, e.g. 'changes/1-a.md'.
        offset: Lines before the text in its file, so an error names the
            file's own line number.

    Returns:
        A dict from section name to its entries, each entry a bullet with its
        continuation lines and no trailing newline.

    Raises:
        ChangelogError: on the first line that breaks the shape, or when a
            section, or the whole text, holds no entry.
    """
    sections: Entries = {}
    section: list[str] | None = None
    entry: int | None = None
    for number, line in enumerate(text.splitlines(), 1):
        at = f'{where}:{number + offset}'
        if not line.strip():
            entry = None
        elif line.startswith('### '):
            name = line[4:].strip()
            if name not in SECTIONS:
                raise ChangelogError(
                    f'{at}: unknown section {name!r}; use one of '
                    + ', '.join(SECTIONS)
                )
            section = sections.setdefault(name, [])
            entry = None
        elif line.startswith('#'):
            raise ChangelogError(f'{at}: only ### headings belong in an entry')
        elif line.startswith('- '):
            if section is None:
                raise ChangelogError(f'{at}: a bullet before any ### heading')
            if line.startswith(TODO):
                raise ChangelogError(f'{at}: fill in the TODO bullet')
            section.append(line)
            entry = len(section) - 1
        elif line[0].isspace() and section is not None and entry is not None:
            section[entry] += '\n' + line
        else:
            raise ChangelogError(
                f"{at}: expected a '- ' bullet or an indented continuation"
            )
    empty = [name for name, entries in sections.items() if not entries]
    if empty or not sections:
        raise ChangelogError(
            f'{where}: no entries under '
            + (', '.join(empty) if empty else 'any section')
        )
    return sections


def fragment_order(name: str) -> tuple[int, int, str]:
    """Sorts fragments by leading number, then name; unnumbered ones last."""
    match = re.match(r'\d+', name)
    return (0, int(match.group()), name) if match else (1, 0, name)


def fragments(root: pathlib.Path) -> list[pathlib.Path]:
    """Lists the fragment files under changes/, in fold order.

    Raises:
        ChangelogError: for a file there that is not a .md fragment.
    """
    folder = root / CHANGES
    if not folder.is_dir():
        return []
    found = []
    for path in folder.iterdir():
        if path.name in NOT_FRAGMENTS:
            continue
        if path.suffix != '.md' or not path.is_file():
            raise ChangelogError(f'{CHANGES}/{path.name}: not a .md file')
        found.append(path)
    return sorted(found, key=lambda p: fragment_order(p.name))


def split_unreleased(text: str) -> tuple[str, str, str]:
    """Splits CHANGELOG text around [Unreleased]'s body.

    Returns:
        The head through the [Unreleased] line, its body, and the rest from
        the next ## heading or link definition on.

    Raises:
        ChangelogError: when there is no [Unreleased] heading.
    """
    start = text.find(UNRELEASED + '\n')
    if start < 0:
        raise ChangelogError(f'{CHANGELOG}: no {UNRELEASED} heading')
    body_at = start + len(UNRELEASED) + 1
    end = UNRELEASED_END.search(text, body_at)
    end_at = end.start() if end else len(text)
    return text[:body_at], text[body_at:end_at], text[end_at:]


def collect(root: pathlib.Path) -> tuple[Entries, list[pathlib.Path]]:
    """Gathers everything waiting: [Unreleased]'s entries, then fragments.

    Returns:
        The merged entries per section, and the fragment paths.

    Raises:
        ChangelogError: naming every source that fails to parse.
    """
    head, body, _ = split_unreleased((root / CHANGELOG).read_text())
    paths = fragments(root)
    sources = [(CHANGELOG, body, head.count('\n'))] if body.strip() else []
    sources += [(f'{CHANGES}/{p.name}', p.read_text(), 0) for p in paths]
    merged: Entries = {}
    errors = []
    for where, source, offset in sources:
        try:
            parsed = parse(source, where, offset)
        except ChangelogError as e:
            errors.append(str(e))
            continue
        for name, entries in parsed.items():
            merged.setdefault(name, []).extend(entries)
    if errors:
        raise ChangelogError('\n'.join(errors))
    return merged, paths


def check(root: pathlib.Path) -> None:
    """Checks everything waiting, as `changelog.py check` does.

    Raises:
        ChangelogError: naming every source that fails to parse and, in a
            repo with changes/README.md, the first line under [Unreleased].
    """
    errors = []
    if (root / CHANGES / 'README.md').exists():
        head, body, _ = split_unreleased((root / CHANGELOG).read_text())
        offset = head.count('\n')
        for number, line in enumerate(body.split('\n'), 1):
            if line.strip():
                errors.append(
                    f'{CHANGELOG}:{offset + number}: an entry under '
                    f'{UNRELEASED}; write it as a {CHANGES}/ fragment'
                )
                break
    try:
        collect(root)
    except ChangelogError as e:
        errors.append(str(e))
    if errors:
        raise ChangelogError('\n'.join(errors))


def render(merged: Entries) -> str:
    """Renders merged entries as Keep a Changelog sections."""
    return '\n'.join(
        f'### {name}\n\n' + '\n'.join(merged[name]) + '\n'
        for name in SECTIONS
        if name in merged
    )


def relink(text: str, version: str, previous: str | None) -> str:
    """Points [Unreleased] past the new tag and adds the release's link.

    Raises:
        ChangelogError: when there is no [Unreleased] link to rewrite.
    """
    match = UNRELEASED_LINK.search(text)
    if not match:
        raise ChangelogError(f'{CHANGELOG}: no [Unreleased]: link to rewrite')
    base = match.group(1)
    if previous:
        target = f'{base}/compare/v{previous}...v{version}'
    else:
        target = f'{base}/releases/tag/v{version}'
    lines = (
        f'[Unreleased]: {base}/compare/v{version}...HEAD\n[{version}]: {target}'
    )
    return text[: match.start()] + lines + text[match.end() :]


def release(root: pathlib.Path, version: str, date: str) -> list[pathlib.Path]:
    """Folds everything waiting into a new section; see the module doc.

    Returns:
        The fragment paths it removed.

    Raises:
        ChangelogError: on a bad version or date, an existing section, a
            parse error or nothing to release. Nothing is written then.
    """
    if not VERSION.match(version):
        raise ChangelogError(f'version {version!r}: want X.Y.Z')
    if not DATE.match(date):
        raise ChangelogError(f'date {date!r}: want YYYY-MM-DD')
    text = (root / CHANGELOG).read_text()
    released = RELEASE_HEADING.findall(text)
    if version in released:
        raise ChangelogError(f'{CHANGELOG} already has a [{version}] section')
    merged, paths = collect(root)
    if not merged:
        raise ChangelogError(
            f'nothing to release: no fragments in {CHANGES}/ and nothing '
            f'under {UNRELEASED}'
        )
    head, _, rest = split_unreleased(text)
    section = f'## [{version}] - {date}\n\n' + render(merged) + '\n'
    previous = released[0] if released else None
    text = relink(head + '\n' + section + rest, version, previous)
    (root / CHANGELOG).write_text(text)
    for path in paths:
        path.unlink()
    return paths


def next_number(root: pathlib.Path) -> int | None:
    """The number the next issue or PR will get, or None without gh."""
    try:
        proc = subprocess.run(  # noqa: S603 (NEWEST is a fixed argv)
            NEWEST,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    newest = proc.stdout.strip()
    if proc.returncode:
        return None
    if newest == 'null':  # no issue or PR yet
        return 1
    return int(newest) + 1 if newest.isdigit() else None


def new(root: pathlib.Path, slug: str, section: str) -> pathlib.Path:
    """Writes the next fragment; see the module doc.

    Returns:
        The path it wrote.

    Raises:
        ChangelogError: on a bad slug or section, no changes/ folder, or a
            fragment already at that path. Nothing is written then.
    """
    if not SLUG.match(slug):
        raise ChangelogError(
            f'slug {slug!r}: want lowercase words joined by hyphens, '
            'starting with a letter'
        )
    if section not in SECTIONS:
        raise ChangelogError(
            f'unknown section {section!r}; use one of ' + ', '.join(SECTIONS)
        )
    if not (root / CHANGES).is_dir():
        raise ChangelogError(
            f'no {CHANGES}/ folder; add {CHANGES}/README.md from '
            "dev-toolchain's templates first"
        )
    number = next_number(root)
    name = f'{number}-{slug}.md' if number else f'{slug}.md'
    path = root / CHANGES / name
    if path.exists():
        raise ChangelogError(f'{CHANGES}/{name} already exists')
    ref = f' (#{number})' if number else ''
    path.write_text(f'### {section}\n\n{TODO}{ref}.\n')
    return path


def main(argv: list[str]) -> int:
    """Runs one subcommand; returns the exit status."""
    command, args = (argv[0], argv[1:]) if argv else ('', [])
    date = dt.date.today().isoformat()
    section = DEFAULT_SECTION
    if command == 'release' and len(args) == 3 and args[1] == '--date':
        date = args[2]
    elif command == 'new' and len(args) == 3 and args[1] == '--section':
        section = args[2]
    elif not (
        (command in ('check', 'show') and not args)
        or (command in ('new', 'release') and len(args) == 1)
    ):
        sys.stderr.write(USAGE + '\n')
        return EX_USAGE
    try:
        if command == 'check':
            check(REPO_ROOT)
        elif command == 'show':
            sys.stdout.write(render(collect(REPO_ROOT)[0]))
        elif command == 'new':
            path = new(REPO_ROOT, args[0], section)
            sys.stdout.write(f'{CHANGES}/{path.name}\n')
            if not path.name[0].isdigit():
                sys.stderr.write(
                    'gh could not name the next number, so it is unnumbered; '
                    'rename it to <PR>-<slug>.md once the PR exists\n'
                )
        else:
            for path in release(REPO_ROOT, args[0], date):
                sys.stderr.write(f'removed {CHANGES}/{path.name}\n')
            sys.stderr.write(f'{CHANGELOG}: added [{args[0]}] - {date}\n')
    except ChangelogError as e:
        sys.stderr.write(f'{e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
