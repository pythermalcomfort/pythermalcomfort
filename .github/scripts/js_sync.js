// Keeps one open issue in jsthermalcomfort that lists the pythermalcomfort PRs
// that may need porting. Called from .github/workflows/js-sync.yml with
// actions/github-script. No AI runs here: the decision comes from the PR's
// labels and the files it touched.
//
// - mode "merge": add one line for a merged PR, unless it is labelled
//   "js: not needed" or touched no model code.
// - mode "release": close the open issue and open the next one, carrying
//   over every unticked line so nothing is lost.

const JS_OWNER = "FedericoTartarini";
const JS_REPO = "jsthermalcomfort";
const TRACKING_LABEL = "pythermalcomfort sync";
const TITLE = "pythermalcomfort changes to port";
const LABEL_NEEDED = "js: needed";
const LABEL_NOT_NEEDED = "js: not needed";

// code that jsthermalcomfort mirrors; plots, docs and tests have no JS twin
const PORTED_PATHS = [
  /^pythermalcomfort\/models\//,
  /^pythermalcomfort\/environment\//,
  /^pythermalcomfort\/psychrometrics\//,
  /^pythermalcomfort\/clothing\//,
  /^pythermalcomfort\/jos3_functions\//,
  /^pythermalcomfort\/_internal\//,
  /^pythermalcomfort\/utilities\.py$/,
  /^pythermalcomfort\/classes_input\.py$/,
  /^pythermalcomfort\/classes_return\.py$/,
];

const INTRO = [
  "This issue lists pythermalcomfort pull requests merged into `development` that may need the same change here.",
  "A workflow in pythermalcomfort adds one line per merged PR, so please do not edit the lines by hand except to tick them.",
  "Tick a line once the change is ported, or once you decide it is not needed and say why in a comment.",
  "When pythermalcomfort publishes a release, this issue is closed and every unticked line moves to the next one.",
].join(" ");

// "- [ ] pythermalcomfort/pythermalcomfort#514 ..." -> unticked items keep their text
const UNTICKED = /^- \[ \] .+$/gm;

function itemLine(pr, unlabelled) {
  const note = unlabelled ? " _(not triaged: no `js:` label at merge)_" : "";
  // a zero-width space after @ and # stops a title from mentioning people in
  // jsthermalcomfort or faking another PR's "owner/repo#N " marker
  const title = pr.title.replace(/[@#]/g, (c) => `${c}\u200b`).replace(/\s+/g, " ").trim();
  return `- [ ] ${pr.base.repo.full_name}#${pr.number} ${title} (@${pr.user.login})${note}`;
}

function issueBody(items, carriedFrom) {
  const parts = [INTRO, ""];
  if (carriedFrom) parts.push(`Unticked lines carried over from #${carriedFrom}.`, "");
  parts.push(...items);
  return parts.join("\n") + "\n";
}

// "- [ ] owner/repo#514 title" -> "owner/repo#514 ", the text that identifies a PR's line
function markerOf(line) {
  return line.slice(6).split(" ")[0] + " ";
}

async function listOpenIssues(js) {
  const issues = await js.paginate(js.rest.issues.listForRepo, {
    owner: JS_OWNER,
    repo: JS_REPO,
    state: "open",
    labels: TRACKING_LABEL,
  });
  return issues.filter((i) => !i.pull_request).sort((a, b) => a.number - b.number);
}

async function findOpenIssue(js) {
  // the oldest open one wins, so a duplicate created by hand does not split the list
  return (await listOpenIssues(js))[0];
}

// the issue line for a merged PR, or null when it needs no porting
async function lineFor({ github, owner, repo, pr, core }) {
  const labels = pr.labels.map((l) => l.name);
  if (labels.includes(LABEL_NOT_NEEDED)) {
    core.info(`#${pr.number} is labelled "${LABEL_NOT_NEEDED}", skipping.`);
    return null;
  }
  const labelledNeeded = labels.includes(LABEL_NEEDED);

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner,
    repo,
    pull_number: pr.number,
  });
  const touchesModels = files.some((f) =>
    PORTED_PATHS.some((re) => re.test(f.filename) || (f.previous_filename && re.test(f.previous_filename))),
  );
  if (!labelledNeeded && !touchesModels) {
    core.info(`#${pr.number} touched no ported code and has no "${LABEL_NEEDED}" label, skipping.`);
    return null;
  }
  return itemLine(pr, !labelledNeeded);
}

async function onMerge({ github, js, context, core, prNumber, dryRun }) {
  const { owner, repo } = context.repo;
  const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
  if (!pr.merged) {
    core.info(`#${prNumber} was closed without merging, nothing to do.`);
    return;
  }

  const issue = await findOpenIssue(js);
  const body = issue ? issue.body || "" : "";

  // GitHub cancels a pending run when a newer one queues behind it, so check
  // every PR merged since the issue opened, not only the one that triggered
  // this run. Ticked lines still contain their marker, so they are not re-added.
  const prs = [pr];
  if (issue) {
    const { data: recent } = await github.rest.pulls.list({
      owner,
      repo,
      state: "closed",
      base: pr.base.ref,
      sort: "updated",
      direction: "desc",
      per_page: 50,
    });
    for (const other of recent) {
      if (other.number !== pr.number && other.merged_at && other.merged_at > issue.created_at) prs.push(other);
    }
  }

  const lines = [];
  for (const candidate of prs) {
    if (body.includes(`${candidate.base.repo.full_name}#${candidate.number} `)) {
      core.info(`#${candidate.number} is already listed in ${issue.html_url}.`);
      continue;
    }
    const line = await lineFor({ github, owner, repo, pr: candidate, core });
    if (line) lines.push(line);
  }
  if (lines.length === 0) return;

  if (!issue) {
    core.info(`No open tracking issue, creating one with:\n${lines.join("\n")}`);
    if (!dryRun) {
      const { data } = await js.rest.issues.create({
        owner: JS_OWNER,
        repo: JS_REPO,
        title: TITLE,
        labels: [TRACKING_LABEL],
        body: issueBody(lines),
      });
      core.notice(`Created ${data.html_url}`);
    }
    return;
  }

  core.info(`Adding to ${issue.html_url}:\n${lines.join("\n")}`);
  if (!dryRun) {
    // re-read the body just before writing, so a line ticked while this run
    // fetched the PRs is not unticked
    const { data: fresh } = await js.rest.issues.get({
      owner: JS_OWNER,
      repo: JS_REPO,
      issue_number: issue.number,
    });
    const freshBody = fresh.body || "";
    const missing = lines.filter((line) => !freshBody.includes(markerOf(line)));
    if (missing.length === 0) return;
    await js.rest.issues.update({
      owner: JS_OWNER,
      repo: JS_REPO,
      issue_number: issue.number,
      body: freshBody.replace(/\s*$/, "\n") + missing.join("\n") + "\n",
    });
  }
}

async function onRelease({ js, core, tag, dryRun }) {
  const open = await listOpenIssues(js);
  const issue = open[0];
  if (!issue) {
    core.info("No open tracking issue, nothing to roll over.");
    return;
  }
  const carried = (issue.body || "").match(UNTICKED) || [];
  core.info(`Closing #${issue.number} for ${tag}, carrying over ${carried.length} unticked line(s).`);
  if (dryRun) {
    core.info(carried.join("\n"));
    return;
  }

  // open the next issue before closing the old one, so a failure never drops the
  // items. A retry after a partial failure reuses the successor it already opened.
  const carriedNote = `carried over from #${issue.number}.`;
  let next = open.find((i) => i.number !== issue.number && (i.body || "").includes(carriedNote));
  if (next) {
    const missing = carried.filter((line) => !next.body.includes(markerOf(line)));
    core.info(`Reusing #${next.number}, adding ${missing.length} line(s) listed since it opened.`);
    if (missing.length > 0) {
      await js.rest.issues.update({
        owner: JS_OWNER,
        repo: JS_REPO,
        issue_number: next.number,
        body: next.body.replace(/\s*$/, "\n") + missing.join("\n") + "\n",
      });
    }
  } else {
    ({ data: next } = await js.rest.issues.create({
      owner: JS_OWNER,
      repo: JS_REPO,
      title: TITLE,
      labels: [TRACKING_LABEL],
      body: issueBody(carried, issue.number),
    }));
  }
  await js.rest.issues.createComment({
    owner: JS_OWNER,
    repo: JS_REPO,
    issue_number: issue.number,
    body: `pythermalcomfort ${tag} is out. ${carried.length} unticked line(s) moved to #${next.number}.`,
  });
  await js.rest.issues.update({
    owner: JS_OWNER,
    repo: JS_REPO,
    issue_number: issue.number,
    title: `${TITLE} (up to ${tag})`,
    state: "closed",
    state_reason: "completed",
  });
  core.notice(`Closed #${issue.number}, opened ${next.html_url}`);
}

module.exports = async ({ github, context, core, getOctokit }) => {
  const js = getOctokit(process.env.JS_SYNC_TOKEN);
  const mode = process.env.MODE;
  const dryRun = process.env.DRY_RUN === "true";
  if (mode === "merge") {
    await onMerge({ github, js, context, core, prNumber: Number(process.env.PR_NUMBER), dryRun });
  } else if (mode === "release") {
    await onRelease({ js, core, tag: process.env.TAG, dryRun });
  } else {
    core.setFailed(`Unknown MODE "${mode}", expected "merge" or "release".`);
  }
};
