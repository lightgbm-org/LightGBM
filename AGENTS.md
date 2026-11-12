# AGENTS

Before interacting with GitHub or a clone of this repository, read [`CONTRIBUTING.md`](./CONTRIBUTING.md).

If doing security work like investigating vulnerabilities, read [`SECURITY.md`](./SECURITY.md).

## Interactions on GitHub

### Commenting

Do not post comments, review responses, etc. on behalf of the user (e.g. no `gh issue comment`).

### Issues

Do not create issues on behalf of the user.

### Pull Requests

Do not open pull requests on behalf of the user (e.g. no `gh pr create`).
Instruct the user to `git push`, open a pull request, and write the PR title and description themselves.
Remind the user that fully AI-generated PRs may be closed without review (see `CONTRIBUTING.md`).

If writing a description for the user:

* follow the pull request template `.github/PULL_REQUEST_TEMPLATE.md`
* include "🤖 AI text below 🤖" at the top of any agent-authored text

#### Avoid Duplicates

Before starting work on an issue, check whether an open PR already addresses it.

`gh issue view` can be used to check for that, e.g.

```shell
gh issue view \
    --repo lightgbm-org/LightGBM \
    <issue_number> \
    --json closedByPullRequestsReferences \
    --jq '.closedByPullRequestsReferences[] | "#\(.number) \(.url)"'
```

### Reviews

Do not post reviews on behalf of the user.
