# maintaining

This document is for LightGBM maintainers.

## Releasing

### Step 1: Put up a Release PR

Create a pull request into `master` which prepares the source code for release.

Copy the description and checklist from the previous release PR (for example: https://github.com/lightgbm-org/LightGBM/pull/6796).

This should usually also include a checklist of other issues and PRs that should be completed for the release,
and the PR should be used to discuss what makes it into the release.

### Step 2: Merge the Release PR

Once the PR is approved, merge it.

Do not merge any other PRs into `master` until the rest of the release is complete.

### Step 3: Wait for a New CI Run on `master`

Wait for all CI runs triggered by the merge to `master` to complete successfully.

These runs build and test the official artifacts that will be attached to the GitHub release and published to package managers.

### Step 4: Create a Release

Navigate to https://github.com/lightgbm-org/LightGBM/releases.

Click "edit" on the draft release that `release-drafter` has created there.

* update the tag and release title to match the version of LightGBM, in the format `v{major}.{minor}.{patch}`
* ensure that tag points at the commit on ``master`` created by merging the release PR

When you're satisfied with the state of the release, click "Publish release".

### Step 5: Upload Artifacts

After creating a release, run the following from the root of the repo to populate it with artifacts.

```shell
# download all artifacts to a local directory
./.ci/downloads-artifacts.sh ${COMMIT_ID}

# attach them to the GitHub release
gh release upload \
    --repo lightgbm-org/LightGBM \
    "${TAG}" \
    ./release-artifacts/*
```

Where:

* `COMMIT_ID` = full commit hash of the commit on `master` corresponding to the release
* `TAG` = the tag for the release (e.g. `v4.6.0`)

### Step 6: Complete All Other Post-merge Release Steps

These include things like publishing to package managers, updating build configs for repackagers like ``conda-forge``, and many other steps.

See the release checklist on the PR for details.
