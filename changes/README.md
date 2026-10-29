# Unreleased changes

One file per change, never an edit to `## [Unreleased]` in `CHANGELOG.md`:
two PRs that both add lines there conflict, and these files cannot.

Name it `<number>-<slug>.md`, the issue or PR number, then a few words
(`119-gh-runners-scan.md`). Write Keep a Changelog sections and their
bullets only, a continuation line indented:

```markdown
### Fixed

- `gh-runners` logs why a scan failed instead of saying nothing was
  registered (#119).
```

Sections: Added, Changed, Deprecated, Removed, Fixed, Security.
`changelog.py check` runs in CI and names any file that breaks this shape,
and any line left under `[Unreleased]`;
`changelog.py show` prints what the next release would add, and
`changelog.py release X.Y.Z` folds every file here into `CHANGELOG.md` and
deletes them. This README stays.
