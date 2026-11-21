# Visual consistency pass — design

Status: **implemented** 2026-11-21 — what changed during the build is in §13
Date: 2026-11-21 (revised the same day after an independent builder review)
Scope: `backend/templates/*.html` (6 files), `backend/app/static/css/*.css` (5 files),
`backend/tools/design_audit.py`, `documentation/DESIGN.md`

---

## 1. Intent

PINE's visual language is in better shape than it looks. Colour is already
tokenised, the six-step type scale is respected everywhere, and the z-index
ladder is honoured in all but one place. What makes the app read as untidy is
not the palette — it is that **the same decision is made many times, slightly
differently, in eleven files**.

Two different failures hide under the word "inconsistency", and they need
different fixes:

**a) The code has drifted from rules `DESIGN.md` already states.** Radii must be
tokens — 26 are literals. Focus rings must be `0 0 0 3px var(--accent-dim)` —
nine rings are 1px, 2px or 2.5px instead. Interactive transitions must be
`0.15s ease` — thirty are not. A raw `z-index` is forbidden — one exists. These
are not design decisions to be made; they are already made, and the code is
behind.

**b) Decisions `DESIGN.md` never made at all.** There is no spacing scale, so 31
different pixel values are in use across 724 declarations. There is no guidance
for elevation, non-button control heights, line-height or letter-spacing, so
each grew its own vocabulary. And six standalone templates with no
`{% extends %}` mean shared furniture — the top bar above all — is written out
five times and has drifted five ways.

The goal is one visual language, enforced by a check rather than by discipline,
with the app looking essentially as it does today.

### Why now

The user's words: the design feels dated and inconsistent. A separate,
structural redesign was explored, mocked up and deliberately parked. This
document covers only the tidiness half — the character of the UI does not
change here.

---

## 2. Goals

- One spacing scale, used everywhere, enforced by an automated check.
- Shared furniture defined once: top bar, toggle switch, colour grid.
- Every rule already written in `DESIGN.md` actually holds in the code.
- `DESIGN.md` extended to cover what it currently leaves open — spacing,
  elevation, rings, control heights, line-height, letter-spacing.
- A reviewer can tell, from one command, whether the app still complies.

## 3. Non-goals

- Changing the character of the UI: density, palette, type scale, radii,
  uppercase micro-labels, speaker fills in the transcript. All stay.
- The structural redesign (project workbench, bottom player, margin codes). It
  was mocked up, discussed and parked. Nothing here forecloses it.
- Dark-theme palette work beyond fixing the two mis-scoped selectors.
- Accessibility beyond what consistency drags in (see §7, "focus").
- Any change to Python, API shapes, transcript/annotation formats or storage.

## 4. Anti-goals

Things that would make this a failure even if every number lined up:

- **Tightening the interface under cover of tidying.** Collapsing `10px → 8px`
  (83 declarations) and `6px → 4px` (75) would do exactly that. The chosen scale
  keeps 6 and 10 for this reason.
- **Trading readable CSS for token purity.** Replacing all 724 spacing literals
  with `var(--space-*)` makes diffs unreviewable and the CSS harder to read at a
  glance, for no visible gain.
- **A rule nobody can check.** If the scale is not machine-verifiable, it decays
  within weeks — which is how the current state arose.
- **Losing a deliberate exception by treating it as drift.** `--btn-radius: 8px`
  against `--radius: 6px` is documented and intended. `save-status.css`'s literal
  `z-index: 2100` is documented and intended. Both stay.

## 5. Constraints

- **No version control.** The working copy is not a git repository. For a change
  touching ~700 declarations this is the dominant risk, and it drives the
  phasing and the rollback plan in §9. This needs the user's decision — see §11.
- **Root font size is user-controlled** (`html { font-size: <setting>px }`), so
  type stays in `rem`. Spacing stays in `px` and does not scale with it — that is
  the existing behaviour and this document does not change it.
- **Six standalone templates, no `{% extends %}`** and no build step. Shared CSS
  can only be shared by moving it into `static/css/` and linking it; the
  templates' own `<style>` blocks remain.
- `backend/app/` must never import ML packages — irrelevant here, but the files
  touched are served by it, so nothing in this work may alter Flask routing or
  template context.
- A reset deletes anything at the repo root not listed in
  `backend/tools/reset_preserve_root.txt`. `documentation` is listed, so this
  file and `DESIGN.md` survive. No new root-level file is introduced.
- **The shared sheets load _after_ each template's `<style>` block.** In all six
  templates the boot style and `fonts.css` / `tokens.css` come first (lines
  10–12), the page's own `<style>` runs to line 828 / 253 / 500 / …, and
  `button-system.css`, `ui-system.css`, `save-status.css` are linked *after* it
  (`main.html:831-833`, `settings.html:255-257`, and so on). At equal
  specificity **the shared sheet wins, not the template.**

  This contradicts both `DESIGN.md` ("linked by all six templates before their
  own `<style>` block") and `ui-system.css`'s own header comment ("Load … BEFORE
  the page's own `<style>` block, so a page can still override a primitive
  locally"). Consequence: an unknown number of template rules that look like
  local overrides are inert — `recording.html:405`'s `.rtab { border-bottom: 2px }`
  loses to `button-system.css:389`'s `border: 1px solid transparent` shorthand,
  so that tab underline does not render today. See §11.2 — this is a user
  decision, not a builder one.

- 760 tests must stay green. Eight test modules read templates.
  **`test_transcript_edit_affordance.py:101-103` pins exact CSS text**, including
  the newline and two-space indent in `'.edit-tools .edit-subaction {\n  display: none;'`
  and the literal `top: calc(100% + 6px)`. Reformatting that rule, or moving it
  into a shared sheet, fails the suite with no visual change. 6px is on the
  scale, so nothing in this plan moves it — but that rule is not to be
  reformatted, and if it ever is, the test changes in the same commit.

---

## 6. Approaches considered

### A. Full tokenisation

Introduce `--space-*` and rewrite all 724 spacing declarations as `var(--space-n)`.
Maximum enforcement: a wrong value becomes unwriteable without inventing a token.

Rejected. The diff is ~700 lines of mechanical substitution across eleven files
with no version control to review it against, and `padding: var(--space-2) var(--space-4)`
is harder to read in a 3,800-line template than `padding: 8px 16px`. The
enforcement it buys is better bought by a check.

### B. Round the literals, no tokens, no check

Snap every value to the scale and stop. Smallest diff, nothing new to learn.

Rejected. Nothing prevents `7px` from reappearing next week. This is precisely
how the current 31 values accumulated — the project already has a design
document stating rules that the code then drifted from.

### C. Hybrid — tokens in shared CSS, literals in templates, enforced by a check

**Chosen.** Tokens are defined in `tokens.css` and used in `button-system.css`,
`ui-system.css` and all new code. Template-local literals are rounded to the
scale but left as literals. Compliance is enforced by extending
`backend/tools/design_audit.py`, which already exists for exactly this purpose
(it compares typographic properties across semantic roles) and already follows
the repo's `check_*.py` convention.

This keeps diffs reviewable, keeps template CSS legible, and makes the rule
checkable. The cost: a literal in a template is only as correct as the last
audit run — accepted, because the audit is one command.

---

## 7. The decisions

### 7.1 Spacing scale (user's choice)

```
2  4  6  8  10        micro   — inside controls: icon gaps, chip padding
12 16 20 24 32        block   — between elements and inside panels
40 48                 page    — screen gutters and empty-state padding
```

Twelve steps in three registers. `2/4/8/12/16/24/32` was considered and
rejected: it moves 48 % of declarations and quietly tightens the UI.

Measured over **615 spacing declarations** — `padding*`, `margin*`, `gap`,
`row-gap`, `column-gap` inside `<style>` blocks and `.css` files, which is
exactly the surface this pass edits. (An earlier count of 724 wrongly included
85 declarations in `style="…"` attributes, which §7.1 then excludes from the
work. The thirteen visible changes are unaffected — all of them live in `<style>`
blocks.)

| Movement | Count | Visible? |
|---|---:|---|
| exact hit, no change | 454 | — |
| 1px (odd → even) | 91 | no |
| 2px | 57 | barely |
| 4px | 13 | yes |

The thirteen visible changes, named:

| File | Line | Change |
|---|---|---|
| `templates/onboarding.html` | 51 | `padding: 28px 0` → `24px 0` |
| `templates/onboarding.html` | 56 | `padding: 0 20px 28px` → `0 20px 24px` |
| `templates/onboarding.html` | 129 | `padding: 52px 48px` → `48px 48px` |
| `templates/onboarding.html` | 155 | `.panel-desc` `margin-bottom: 28px` → `24px` |
| `templates/onboarding.html` | 158 | `.opt-grid` `margin-bottom: 28px` → `24px` |
| `templates/onboarding.html` | 234 | `margin-bottom: 28px` → `24px` |
| `templates/onboarding.html` | 240 | `.field-stack` `margin-bottom: 28px` → `24px` |
| `templates/onboarding.html` | 445 | `.syscheck-list` `margin-bottom: 28px` → `24px` |
| `templates/settings.html` | 66 | `.content` `padding: 28px` → `24px` |
| `templates/settings.html` | 70 | `.section` `margin-bottom: 28px` → `24px` |
| `templates/main.html` | 129 | `.search-in` `padding: 0 28px 0 30px` → `0 24px 0 32px` |
| `templates/main.html` | 233 | `padding: 28px 16px` → `24px 16px` |
| `templates/manage_tags.html` | 116 | `padding: 0 28px 0 9px` → `0 24px 0 8px` |

Eleven of the thirteen are the same decision — **the 28px block rhythm becomes
24px** — and it lands almost entirely in onboarding and settings. Those two
screens get a side-by-side look before and after; the rest need none.

`main.html:3035` carries `padding: 40px` inside a JavaScript string that builds
a fatal-error screen (`document.body.innerHTML = '<div style="…">'`). 40px is on
the scale, so it does not move, but it is listed here because a text-level
find-and-replace over spacing values would reach into JS strings. **Spacing
edits are confined to `<style>` blocks and `.css` files; `style="…"` attributes
and JS string literals are left alone in this pass.**

### 7.2 Elevation and rings

`box-shadow` currently carries three unrelated jobs. Splitting them first, the
real numbers are much smaller than a raw count suggests:

| Job | Current | Target |
|---|---|---|
| Elevation | 18 via `--shadow-*`, **5 literal** | all via `--shadow-sm/md/lg` |
| Ring — **mostly not focus rings** | 9 literals at 1px, 2px, 2.5px, 3px | **reclassify, do not convert** — see below |
| Inset rule / underline | 3 literals | `--rule-inset`, or left local with a comment |
| Reset (`none`, transparent) | 8 | unchanged |

The five literal elevations are near-misses of existing tokens
(`0 8px 24px rgba(0,0,0,.18)` against `--shadow-md`'s `.14`;
`0 1px 3px rgba(0,0,0,.08)` against `--shadow-sm`'s `.06`) and snap to them.
Two are genuinely heavier (`0 16px 48px rgba(0,0,0,.65)`,
`0 10px 24px rgba(0,0,0,.22)`) — they go to `--shadow-lg`, which is a visible
change on those two surfaces and is listed for review.

**Correction to the earlier assessment:** "20 hand-written shadows" counted
rings, inset rules and resets as shadows. The honest figure is five.

**Second correction — the nine rings are not focus rings.** Five are
`0 0 0 1px var(--border)` on `[contenteditable]:focus` in `main.html` — hairline
outlines standing in for a border, including the inline-edit affordance at
`main.html:450`. One is the playhead knob's halo (`recording.html:282`,
`0 0 0 2.5px var(--accent)`). Three are content state — the playing utterance
and the pending selection. One is a cut-out against the card
(`button-system.css:581`). Converting them to `0 0 0 3px var(--accent-dim)`
would put a green halo on the transcript's edit affordance and fatten the
scrubber knob.

So: **none of the nine becomes a focus ring.** They are catalogued, given
width-only tokens (`--ring-hairline: 0 0 0 1px`, `--ring-focus: 0 0 0 3px`) that
compose with whatever colour the use demands, and otherwise left as they are.
`DESIGN.md`'s 3px accent-dim rule applies to *focus*, which these are not.

Composite shadows get their own bucket: `recording.html:314, 323` layer an
inset hairline, a ring and a glow on `.utt.playing`; `tags.html:185-186` use
`inset 0 ±2px 0 var(--accent)` as drop indicators. The check (§8) must accept
any value that contains a token or an `inset`, or it will flag these forever.

**Third correction — the two "heavy" shadows.** `button-system.css:761` and
`:767` are a rest/hover **lift pair** on `.back-to-rec-btn`
(`0 6px 18px` → `0 10px 24px`), not two elevations; sending the hover state to
`--shadow-lg` (`0 20px 60px`) turns a 4px lift into a 14px one, so the pair
stays as it is pending a hover-lift token. `recording.html:194`'s `.sel-popup`
is a popover, and `tokens.css` assigns popovers to `--shadow-md`, not `-lg` —
corrected to `--shadow-md`.

### 7.3 Duplicated components

| Component | Today | Target |
|---|---|---|
| `.top-bar`, `.tb-logo`, `.tb-name`, `.tb-sub`, `.tb-sep`, `.tb-logo-icon` | defined in 5 templates. Drift is **smaller than first reported**: `.tb-sep` is `margin: 0 14px` in all five, and main's `6px` is `.tb-sep.compact`, a deliberate modifier. Real differences: `flex-shrink: 0` present in three of five, and `main.html`'s `.top-bar` is the only one without `z-index: var(--z-sticky)` | one `topbar.css`, linked by all five. Note it gives main a stacking context it does not have today |
| `.sw` (toggle) | **30×17 with border in main + recording; 36×20 without border in settings**; knob 11px vs 16px; travel 15px vs 18px; transition 0.15s vs 0.2s | one definition in `ui-system.css` at **36×20, settings' version** (user's decision — an easier target to hit). **The switches in main and recording grow.** Convenient given the cascade: a shared `.sw` wins over both templates' local rules automatically |
| `.color-grid` | gap `5px` in recording, `6px` in tags and manage_tags | one definition, `6px` |
| `.shell` | 5 templates, two variants — and onboarding's is a **different layout** (`display: flex` in a row, no top bar), while settings' omits `overflow: hidden` | **dropped from scope.** Three lines, two genuinely different layouts, and with the cascade running as it does a shared `.shell` would silently override onboarding's and collapse the wizard. Not worth the risk |
| 33 shared classes | overridden locally in templates (`.btn-outline`, `.export-btn`, `.proj-pill`, `.rtab`, `.field`, …) | override only where the override is intentional and commented; otherwise delete |

In all, 48 classes are defined in two or more templates and 37 of them have
drifted.

### 7.4 Rule compliance (rules `DESIGN.md` already states)

| Rule | Violations | Where |
|---|---:|---|
| Never write a raw `z-index` | 1 | `recording.html:91` (`z-index: 40` → `var(--z-dropdown)`) |
| Never hardcode radii | 26 | `2px` ×8, `3px` ×15, `5px`, `7px`, `8px` ×1 — mostly drag indicators and small markers; map to `--radius-sm` or a new `--radius-xs: 2px` |
| Never hardcode colours in component CSS | 30 (21 unique; 9 are `#fff`) | `#fff` → `--accent-fg` or `--card`; `#080810` (video stage) and the `hsla(48 …)` selection highlights need real tokens, not a blanket swap |
| Focus ring is `0 0 0 3px var(--accent-dim)` | 9 rings at other thicknesses | see §7.2 |
| Interactive transitions are `0.15s ease` | 30 of 117 | `0.1`, `0.12`, `0.14`, `0.18`, `0.2`, `0.25`, `0.3`, `0.4`. Non-interactive animations (spinners, pulses) are exempt and stay. |
| Scope dark theme as `:root[data-theme="dark"]` | 2 | `recording.html:383`, `recording.html:395` — a bare `[data-theme="dark"]` also matches the theme-picker buttons |

### 7.5 Vocabulary that `DESIGN.md` does not yet cover

- **Line-height** — 11 values today. Target four, named: `1` (single-line
  controls), `1.3` (headings), `1.5` (UI text), `1.65` (long-form and transcript).
- **Letter-spacing** — 13 values in two unit systems (`0.08em` beside `-0.3px`).
  Target three: `0` (default), `0.08em` (`--label-tracking`, existing),
  `0.14em` (the PINE wordmark). The six negative px trackings on headings are
  dropped; at 10–20px type they are below the threshold of visibility and they
  are the only place in the app sizing type-adjacent values in px.
- **Control heights** — 14 distinct values between 16 and 56px. Buttons and
  icon-buttons are already tokenised (30/34/38, 20/22/28). Untokenised: fields,
  the search input, switches, the top bar. Target: `--h-field: 32px`,
  `--h-field-sm: 28px`, `--h-topbar: 48px`, switch via the component.

### 7.6 Focus states

`focus-visible` appears in `button-system.css` (5 rules) and once each in main,
recording and settings. **tags.html, manage_tags.html and onboarding.html have
none at all.** Fields get a ring in addition to today's border-colour change, so
a keyboard user can see where they are on every screen.

Out of scope but recorded: 53 `div`/`span` elements carry `onclick` and cannot
take focus at all (against 150 real `<button>`s). Converting them is a separate
task — it touches behaviour, not appearance.

### 7.7 `DESIGN.md` itself

The Layout section states **"Top bar: 20px height"**, while `.top-bar` is `48px`
in all five templates that define it. `--top-bar-h: 20px` does exist in
`main.html`'s `:root`, but it is an alignment rule for the right-hand header,
not the bar's height. The document and the code disagree and one of them is
wrong; the builder resolves this by reading `main.html`'s layout, and documents
whichever is true. **This is the one place where the audit found the document,
not the code, to be the likely error.**

`DESIGN.md` gains: the spacing scale, the elevation/ring split, control heights,
line-height, letter-spacing, and a line pointing at the audit command.

---

## 8. The check

Extend `backend/tools/design_audit.py` with a second report — the existing
role-based typography comparison is untouched:

- spacing literals outside the scale, as `file:line`
- `border-radius`, colour and `z-index` literals outside tokens
- `transition` durations other than `0.15s` on interactive properties
- `box-shadow` values that are neither a token nor a recognised ring
- classes defined in more than one template
- bare `[data-theme="dark"]`

Exit code non-zero when anything is found, so it can later be wired into CI. It
is a report, not a gate, on first landing — it will have findings until the last
phase is done.

A unit test asserts the audit runs and that the scale list matches the one in
`tokens.css`, so the two cannot drift apart.

**Two audit tools exist; they do different jobs.** `backend/tools/design-audit.js`
(28KB, Playwright, drives the live routes) already collects *computed*
`border-radius`, `box-shadow`, `transition`, `padding-*`, `margin-*`, `gap` and
compares them across pages. It is left untouched. The new reports go in the
Python tool because they are about *source literals*, which is a static
question — but note the Python tool today globs only `templates/*.html` and must
be extended to read `static/css/` as well. Neither tool can resolve which rule
wins across files; only the JS one, working on computed style, could. That is
why §11.2 matters.

**"Green" needs a definition, or the check decays like the rules it replaces.**
The exemption list is explicit and lives beside the check:

- `--btn-radius: 8px` against `--radius: 6px` — documented, intended
- `save-status.css`'s literal `z-index: 2100` — documented, intended
- non-interactive animations (spinners, pulses) at any duration
- `style="…"` attributes and JS string literals (out of scope this pass)
- spacing values above 48px: `tags.html`'s `padding-bottom: 60px` and
  `manage_tags.html`'s `80px` are scroll tails, and negative margins
  (`-5px`, `-14px`, `-24px`) are offsets, not rhythm — all exempt, each with a
  comment naming why
- `top` / `left` / `right` / `bottom` and `background-position` are **not** part
  of the spacing scale. They are offsets, often paired to a padding, and two
  such pairs must be checked by hand when their padding moves:
  `main.html:123` `.s-ico { left: 9px }` sits inside `.search-in`
  (padding-left 30 → 32), and `manage_tags.html:116` is a `<select>` whose
  `padding-right: 28px → 24px` would cut clearance to its 12px chevron at
  `background-position: right 9px center` from ~7px to ~3px. The chevron offset
  moves with the padding.

---

## 9. Phasing and rollback

Phases land one at a time, each independently reversible, each ending with
`cd backend && pytest` and `python -m ruff check .`.

| # | Phase | Visual effect |
|---|---|---|
| 0 | Snapshot; capture the **before** screenshot set (§9.1); build the inert-rule inventory (§11.2); add the audit's new report and its test | none |
| 1 | Add tokens to `tokens.css` (`--space-*`, `--ring*`, `--lh-*`, `--track-*`, `--h-*`, `--radius-xs`) | none — nothing consumes them yet |
| 2 | Rule compliance: z-index, radii, colours, rings, transitions, `[data-theme]` | near-none; the two heavy shadows change |
| 3 | Deduplicate: top bar into `topbar.css`, `.sw`, `.color-grid`; delete only the overrides the §11.2 inventory proves inert or unintended | **the toggles in main and recording grow to 36×20**; main's top bar gains a stacking context. `.shell` is out of scope (§7.3) |
| 4 | Spacing scale across templates and shared CSS | 167 declarations move; 13 visible, 11 of them `28px → 24px` |
| 5 | Line-height, letter-spacing, control heights, `focus-visible` on the three screens that lack it | small, broad |
| 6 | Update `DESIGN.md`; resolve the 20px/48px top-bar contradiction | docs only |

### 9.1 Screenshot gate

Every phase from 1 onward ships with a before/after screenshot pair per affected
screen, so the "visual effect" column above is verifiable rather than asserted.

The project already has the pieces:

- `backend/tools/screenshots/` holds light/dark pairs and already uses an
  `-after` suffix (`settings-after-light.png` beside `settings-light.png`), so
  the naming convention exists and is followed, not invented.
- `backend/tools/design-audit.js` drives the live app headless through Playwright
  with route discovery (`STATIC_ROUTES` is `/` and `/settings`, the rest found at
  runtime). Capture is added there rather than in a new tool.

What the builder adds: theme switching, which `design-audit.js` does not do
today — each route is captured with `data-theme` set to light and to dark.

Coverage: both themes × the six screens, plus the four user-selectable root font
sizes (12 / 13 / 15 / 17px, `settings.html:316-319`) on the two screens where
control heights change, since a fixed height clips at the larger sizes (§7.5).
The **before** set is captured in Phase 0, against untouched code.

Screenshots are the gate, not decoration: a phase is not finished until its
pairs have been looked at and the differences are the ones this document names.

**Rollback — the first draft of this section was wrong.** It proposed copying
files into `backups/`, claiming that directory exists and is reset-safe. All
three claims are false: `backups/` does not exist in this working copy; its
location is a user setting (`_backup_dir()` in
`backend/app/services/backup_service.py:46-53` reads `backup_path`, defaulting
to `ROOT_DIR/backups`) and it holds the user's research backups; and `backups`
is **not** in `backend/tools/reset_preserve_root.txt`, so a reset deletes it
(`reset.command:152` runs `rm -rf` over everything unlisted).

Corrected: snapshots go to a timestamped folder in the OS temp directory, the
pattern `backup_service._external_safety_backup_dir()` already uses — never into
the user's backup folder. This is still materially weaker than a branch, which
is why §11.1 asks about `git init` first and why that question is closer to
blocking than the first draft implied.

**Reviewing `onboarding.html` is not free.** `/` renders onboarding only while
`onboarding_complete != 'true'` (`backend/app/__init__.py:531`), and the other
routes redirect away while it is false — the wizard and the app are mutually
exclusive states of one install. Eleven of the thirteen visible spacing changes
land there.

`AGENTS.md` §3 forbids running the reset scripts without an explicit request, so
the reset is not the route. Instead: copy the repo to a scratch directory, point
it at its own empty `backend/data/pine.db`, and let onboarding run there;
`goStep(n)` at `onboarding.html:940` reaches individual steps once the page is
up. The user's install is never touched. `settings.html` carries the other two
changes and needs no special setup.

---

## 10. Left to the builder

- Exact token names, within the conventions already in `tokens.css`.
- Whether `--radius-xs: 2px` is worth adding or the eight 2px radii should become
  `--radius-sm` (4px) — a judgement call on drag indicators and 6px dots.
- Which of the 33 local overrides are intentional. The test: an override that
  changes geometry is suspect; one that changes only placement (`flex: 1`,
  `margin-left: auto`) is fine and stays, with a comment.
- Whether `topbar.css` is a new file or a section of `ui-system.css`. A new file
  is cleaner but adds a `<link>` to six templates; `ui-system.css` is already
  linked everywhere.
- The precise hsla tokens for `recording.html`'s selection highlights.

### 10.1 Calls made in response to the review

These were settled without going back to the user. They are listed so they can
be checked while reading, and overridden cheaply if any is wrong.

- **Acceptance criterion for Phases 2–3: today's rendering wins.** Where a
  template rule is inert because the shared sheet beats it, deleting the rule is
  correct and we do *not* resurrect its apparent intent. `recording.html:405`'s
  2px tab underline has not painted in a long time; restoring it would be an
  unrequested visual change. It is recorded as a separate follow-up, not fixed
  here.
- **`.shell` is out of scope** (§7.3) — two genuinely different layouts, three
  lines of CSS, and real risk to the onboarding wizard.
- **The nine rings are reclassified, not converted** (§7.2).
- **`.sel-popup` goes to `--shadow-md`**, per `tokens.css`'s own stated role for
  popovers, not `--shadow-lg`. The `.back-to-rec-btn` rest/hover lift pair is
  left exactly as it is until there is a hover-lift token to put it in.
- **Control heights use `min-height`, never `height`**, because the root font
  size is user-selectable up to 17px and a fixed height clips. `main.html:39`'s
  existing `--search-in-h: 28px` folds into `--h-field-sm` rather than sitting
  beside it.
- **The new static checks go in `design_audit.py`**, extended to read
  `static/css/` as well as `templates/`. `design-audit.js` keeps its computed-
  style job and is touched only to add screenshot capture (§9.1).
- **Spacing scope is `padding*`, `margin*`, `gap`, `row-gap`, `column-gap`
  only.** `top`/`left`/`right`/`bottom` and `background-position` are offsets,
  not rhythm, and stay out — with the two paired cases in §8 checked by hand.
- **The check has an explicit exemption list** (§8) so that "green" is a state
  the app can actually reach.
- **Onboarding is reviewed on a scratch copy of the repo** with its own empty
  database — never by running a reset, which `AGENTS.md` §3 forbids without an
  explicit request.

## 11. Decisions taken

1. **No git.** The repository is not initialised and will not be. Rollback is
   the temp-directory snapshot in §9: before each phase, the files that phase
   touches are copied to a timestamped folder, and reverting is a copy back.
   Consequence the builder must respect: **a phase is the smallest revertible
   unit.** There is no per-file history, so phases stay small and land one at a
   time, each verified before the next begins.

2. **The stylesheet link order is investigated before it is decided.**
   Phase 0 produces a machine-readable inventory of every template rule that
   loses to a shared sheet today — selector, file, line, the rule that beats it,
   and whether the loss is visible. Only with that list in hand do we choose
   between leaving the order, moving the links, or correcting the documents.
   Until then, §7.3's "33 local overrides" is treated as an unverified figure
   and no override is deleted.

3. **The toggle is 36×20** — settings' version, the easier target. The switches
   in `main.html` and `recording.html` grow to match, along with the 16px knob
   and 18px travel. This is a visible change on two screens and is covered by
   the screenshot gate below.

4. **All phases, 0 through 6**, with a before/after screenshot gate (§9.1) so
   each phase can be checked by eye rather than taken on trust.

### Correction carried into §7.3

Decision 3 reverses the draft's choice of 30×17. The draft also claimed the
cascade *forced* the smaller size; it does the opposite — because shared CSS
loads last, a shared `.sw` at 36×20 overrides both templates' local rules
without touching them.

## 12. Where the review changed this document

An independent read of the draft found six factual errors, since corrected:
the `backups/` rollback scheme (§9), the claim that no test pins CSS text (§5),
the `.tb-sep` drift example (§7.3), the spacing totals (§7.1), the treatment of
the nine rings as focus rings (§7.2), and the `--shadow-lg` mapping for a hover
pair and a popover (§7.2). It also surfaced the stylesheet link order (§5,
§11.2), which neither the draft nor `DESIGN.md` had right.

The thirteen named visible changes in §7.1 survived the review unchanged.

## 13. Implementation record (2026-11-21)

All phases 0–6 landed the same day. `python backend/tools/design_audit.py --rules`
reports no findings; the suite is unchanged apart from the new
`tests/test_design_audit.py` (the four `test_tier2.py` GPU-detection failures
predate this work and are platform-specific — no CUDA check on Apple Silicon).

### Facts corrected while building

- **There is no `.rtab` underline.** `recording.html:405` set
  `border-bottom: 2px solid transparent`, and no rule ever coloured it. The rule
  was dead, not a lost design; the follow-up in §10.1 is void.
- **`DESIGN.md` did not misstate the link order** — its sentence was about
  `tokens.css`, which does load first. `ui-system.css`'s header comment was the
  wrong one. `DESIGN.md` was nevertheless wrong in four other places (front
  matter `accent-fg`, `font-body`, dark `accent-bd-solid`; the 20px top bar), not
  one.
- **Rings: 5 + 1 + 3 + 1 = 10, not 9**, and an eleventh was missed:
  `main.html` `.seg-stepper:focus-within` is a real focus ring drawn at 2px. It
  now uses `--btn-focus-ring` (user's decision).
- **`manage_tags.html` is not rendered by any route** —
  `/project/<id>/tags/manage` redirects to `tags?edit=1`. Its edits are kept
  consistent but are invisible; one of the "thirteen visible changes" is not.
- **Counts:** 615 is the number of spacing *values* (558 declarations); 161
  values moved, not 167. Radius literals were 23 in `<style>` blocks, plus nine
  `calc(var(--radius) + 4px)` that equal `--radius-lg` and now use it.

### Decisions taken during the build (user)

1. **Link order — moved.** Shared sheets now load before every template's
   `<style>`. The Phase 0 inventory (computed cascade, headless Chrome,
   `documentation/2026-11-21-inert-rule-inventory.json`) found 19 template rules
   that lost by link order with a different value, plus two more on elements JS
   creates. Each was resolved by §10's test: geometry drift deleted (the shared
   value already rendered), intent kept. Net visible effect of the move: the
   search field's clear button now hides when the field is empty, and Settings'
   font buttons preview their own faces — two bugs the old order caused.
2. **Spacing ties round up** (14→16, 18→20, 22→24, odd values to the even step
   above), so the pass never tightens; 28→24 stays as §7.1 decided.
3. **Literal shadows snap to tokens** even where the dark-theme alpha makes the
   change visible (the toast). `.sel-popup` is lighter (`--shadow-md`).
4. **`.seg-stepper` focus ring 2px → 3px.**

### Builder calls

- Top bar, switch and colour grid went into `ui-system.css` (§6–8) rather than a
  new `topbar.css`: it was already linked everywhere. The archive toggle, an
  identical copy in main and recording, moved into `button-system.css`.
- `--radius-xs: 2px` was added for marks in running text and thin indicators.
- Line height snaps to the nearest of 1 / 1.3 / 1.5 / 1.65 (ties up); tracking
  to 0 except uppercase labels (`--label-tracking`) and the wordmark. The audit
  checks both, which §8 had not asked for.
- Every text field gets the focus ring from one `:where(...)`-scoped rule in
  `ui-system.css`, which covers the three screens that had no `focus-visible`.
  Settings' radio focus moved from `outline` to the ring.
- `::highlight(pinePendingTagSel)` keeps literal colours (exempt in the audit):
  `var()` inside highlight pseudos is not resolved reliably across engines. The
  `<mark>` fallback uses page tokens.
- Screenshots were taken with headless Chrome against a seeded scratch copy of
  the repo (pristine and current side by side, motion frozen, deterministic to
  the pixel), not by extending `design-audit.js` — Playwright is not installed
  here. `design-audit.js` is untouched.

### Export dialog (done after the first pass)

The export dialog was copied into main and recording and had drifted: 8px vs
10px rows, `--fg` vs `--muted-fg` labels. Its option rows and format picker now
live once in `ui-system.css` (§3, §9), matching the Export defaults rows in
Settings — 10px rows, `--fg` labels — so main's rows grew 2px and recording's
labels darkened (find-and-replace shares the rows). Main keeps its own
project/selection tabs and recording checklist.

### manage_tags.html (deleted after the first pass)

Nothing rendered it: `/project/<id>/tags/manage` redirects to `tags?edit=1`, and
that redirect stays so old links keep working (`test_api_projects.py` covers
it). With it went its audit exemption (the 80px scroll tail) and its
`.color-pop` duplicate. Five templates remain.

### Left for follow-up

- The `.back-to-rec-btn` lift pair waits for a hover-lift token.
- 53 `div`/`span` elements with `onclick` still cannot take focus (§7.6).
