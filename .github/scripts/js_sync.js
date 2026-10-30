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
  return `- [ ] ${pr.base.repo.full_name}#${pr.number} ${pr.title} (@${pr.user.login})${note}`;
}

function issueBody(items, carriedFrom) {
  const parts = [INTRO, ""];
  if (carriedFrom) parts.push(`Unticked lines carried over from #${carriedFrom}.`, "");
  parts.push(...items);
  return parts.join("\n") + "\n";
}

async function findOpenIssue(js) {
  const issues = await js.paginate(js.rest.issues.listForRepo, {
    owner: JS_OWNER,
    repo: JS_REPO,
    state: "open",
    labels: TRACKING_LABEL,
  });
  // the oldest open one wins, so a duplicate created by hand does not split the list
  return issues.filter((i) => !i.pull_request).sort((a, b) => a.number - b.number)[0];
}

async function onMerge({ github, js, context, core, prNumber, dryRun }) {
  const { owner, repo } = context.repo;
  const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
  if (!pr.merged) {
    core.info(`#${prNumber} was closed without merging, nothing to do.`);
    return;
  }

  const labels = pr.labels.map((l) => l.name);
  if (labels.includes(LABEL_NOT_NEEDED)) {
    core.info(`#${prNumber} is labelled "${LABEL_NOT_NEEDED}", skipping.`);
    return;
  }
  const labelledNeeded = labels.includes(LABEL_NEEDED);

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner,
    repo,
    pull_number: prNumber,
  });
  const touchesModels = files.some((f) =>
    PORTED_PATHS.some((re) => re.test(f.filename) || (f.previous_filename && re.test(f.previous_filename))),
  );
  if (!labelledNeeded && !touchesModels) {
    core.info(`#${prNumber} touched no ported code and has no "${LABEL_NEEDED}" label, skipping.`);
    return;
  }

  const line = itemLine(pr, !labelledNeeded);
  const marker = `${pr.base.repo.full_name}#${pr.number} `;
  const issue = await findOpenIssue(js);

  if (!issue) {
    core.info(`No open tracking issue, creating one with:\n${line}`);
    if (!dryRun) {
      const { data } = await js.rest.issues.create({
        owner: JS_OWNER,
        repo: JS_REPO,
        title: TITLE,
        labels: [TRACKING_LABEL],
        body: issueBody([line]),
      });
      core.notice(`Created ${data.html_url}`);
    }
    return;
  }

  const body = issue.body || "";
  if (body.includes(marker)) {
    core.info(`#${prNumber} is already listed in ${issue.html_url}.`);
    return;
  }
  core.info(`Adding to ${issue.html_url}:\n${line}`);
  if (!dryRun) {
    await js.rest.issues.update({
      owner: JS_OWNER,
      repo: JS_REPO,
      issue_number: issue.number,
      body: body.replace(/\s*$/, "\n") + line + "\n",
    });
  }
}

async function onRelease({ js, core, tag, dryRun }) {
  const issue = await findOpenIssue(js);
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

  // open the next issue before closing the old one, so a failure never drops the items
  const { data: next } = await js.rest.issues.create({
    owner: JS_OWNER,
    repo: JS_REPO,
    title: TITLE,
    labels: [TRACKING_LABEL],
    body: issueBody(carried, issue.number),
  });
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
