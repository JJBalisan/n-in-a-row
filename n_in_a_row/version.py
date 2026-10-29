"""Names the checkout a program runs from. Copy it into a repo and import it.

The house deploys by ``git pull`` (installed scripts are symlinks into the
checkout), so the only honest version is what git says about the working
tree at run time. Every entry point logs it first, so a log can always be
matched to the code that wrote it, including an unmerged branch left
checked out or local edits (``-dirty``). The line goes to the program's log
or to stderr, never to stdout, which belongs to the program's output.

Releases are annotated tags vX.Y.Z (README, "Releasing"), so
``repo_version()`` returns ``v1.2.0-0-gabc1234`` on a release,
``v1.2.0-3-gdef5678`` three commits past it, ``abc1234`` before the first
tag, with ``-dirty`` appended when the tree has local changes, or
``unknown`` when git cannot answer (no git, not a repository, a timeout).

Copying it: put the file beside the program's modules (``lib/``, ``ci/``, a
package) and import it as a module. Here it lives one directory below the
repo root; edit REPO_ROOT when the copy lives at the root or deeper. Then:

    import version
    ...
    logging.info('<repo> %s', version.repo_version())  # the first line

REPO_VERSION, when set in the environment, is returned as is; tests pin it
so their transcripts stay stable.
"""

import os
import pathlib
import subprocess

# The checkout: this file resolved through symlinks, then up one level. Add
# a `.parent` per extra directory when the copy lives deeper.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Absolute: launchd and cron give jobs a minimal PATH, and /usr/bin/git on
# this Mac resolves through a full Xcode, so no install dialog hangs a job.
GIT = '/usr/bin/git'
TIMEOUT_SECONDS = 60


def repo_version() -> str:
    """The version of the checkout this file lives in.

    Returns:
        REPO_VERSION when set, else the git describe string, else 'unknown'.
    """
    pinned = os.environ.get('REPO_VERSION')
    if pinned:
        return pinned
    argv = [GIT, '-C', str(REPO_ROOT), 'describe']
    argv += ['--tags', '--long', '--dirty', '--always']
    try:
        result = subprocess.run(  # noqa: S603 (fixed argv, no shell)
            argv,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    text = result.stdout.strip()
    if result.returncode or not text:
        return 'unknown'
    return text
