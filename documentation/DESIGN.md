---
colors:
  light:
    bg: "hsl(220, 20%, 97%)"
    fg: "hsl(220, 25%, 10%)"
    card: "hsl(0, 0%, 100%)"
    muted: "hsl(220, 15%, 94%)"
    muted-fg: "hsl(220, 10%, 46%)"
    accent: "hsl(152, 60%, 42%)"
    accent-fg: "#fff"
    accent-dim: "hsla(152, 60%, 42%, 0.10)"
    accent-bd: "hsla(152, 60%, 42%, 0.25)"
    accent-on-dim: "hsl(152, 60%, 27%)"
    accent-bd-solid: "hsl(152, 60%, 27%)"
    border: "hsl(220, 15%, 88%)"
    secondary: "hsl(220, 15%, 92%)"
    sunken: "hsl(220, 20%, 95%)"
    destr: "hsl(0, 72%, 51%)"
    destr-dim: "hsla(0, 72%, 51%, 0.10)"
    destr-bd: "hsla(0, 72%, 51%, 0.25)"
    destr-fg: "#fff"
    warn: "hsl(38, 90%, 55%)"
    warn-dim: "hsla(38, 90%, 55%, 0.10)"
    warn-bd: "hsla(38, 90%, 55%, 0.25)"
  dark:
    bg: "hsl(220, 20%, 7%)"
    fg: "hsl(220, 15%, 93%)"
    card: "hsl(220, 16%, 11%)"
    muted: "hsl(220, 14%, 14%)"
    muted-fg: "hsl(220, 10%, 54%)"
    accent: "hsl(152, 55%, 45%)"
    accent-fg: "#fff"
    accent-dim: "hsla(152, 55%, 45%, 0.12)"
    accent-bd: "hsla(152, 55%, 45%, 0.28)"
    accent-on-dim: "hsl(152, 55%, 45%)"
    accent-bd-solid: "hsla(152, 55%, 45%, 0.55)"
    border: "hsl(220, 16%, 17%)"
    secondary: "hsl(220, 14%, 16%)"
    sunken: "hsl(220, 18%, 9%)"
    destr: "hsl(0, 62%, 60%)"
    destr-dim: "hsla(0, 62%, 60%, 0.12)"
    destr-bd: "hsla(0, 62%, 60%, 0.28)"
    warn: "hsl(38, 85%, 60%)"
    warn-dim: "hsla(38, 85%, 60%, 0.10)"
    warn-bd: "hsla(38, 85%, 60%, 0.25)"

typography:
  font-body: "'Inter', sans-serif"
  font-mono: "'JetBrains Mono', monospace"
  scale:
    xs: "0.769rem"
    sm: "0.846rem"
    md: "0.923rem"
    lg: "1rem"
    title: "1.077rem"
    page: "1.538rem"
  line-height:
    control: "1"
    heading: "1.3"
    ui: "1.5"
    prose: "1.65"
  letter-spacing:
    default: "0"
    label: "0.08em"
    wordmark: "0.14em"

spacing: [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]

layout:
  radius-xs: "2px"
  radius-sm: "4px"
  radius: "6px"
  radius-lg: "10px"
  top-bar: "48px"
  right-header-pad-top: "20px"
  left-header-pad-y: "6px"
  left-header-gap: "6px"
  field-h: "32px"
  field-h-sm: "28px"

components:
  button:
    radius: "8px"
    gap: "6px"
    height:
      sm: "30px"
      md: "34px"
      lg: "38px"
    padding-x:
      sm: "10px"
      md: "14px"
      lg: "16px"
    icon-size:
      xs: "20px"
      sm: "22px"
      md: "28px"
    font-size:
      sm: "0.846rem"
      md: "0.923rem"
      lg: "1rem"
    focus-ring: "0 0 0 3px var(--accent-dim)"
    transition: "background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease, transform 0.15s ease"
  button-primary:
    backgroundColor: "hsl(152, 60%, 42%)"
    textColor: "#ffffff"
  button-primary-dark:
    backgroundColor: "hsl(152, 55%, 45%)"
    textColor: "#ffffff"
  button-secondary:
    backgroundColor: "hsl(220, 15%, 92%)"
    textColor: "hsl(220, 25%, 10%)"
  button-secondary-dark:
    backgroundColor: "hsl(220, 14%, 16%)"
    textColor: "hsl(220, 15%, 93%)"
  button-destructive:
    backgroundColor: "hsl(0, 72%, 51%)"
    textColor: "#ffffff"
  button-destructive-dark:
    backgroundColor: "hsl(0, 62%, 60%)"
    textColor: "#ffffff"
  button-muted:
    backgroundColor: "hsl(220, 15%, 94%)"
    textColor: "hsl(220, 10%, 46%)"
  button-muted-dark:
    backgroundColor: "hsl(220, 14%, 14%)"
    textColor: "hsl(220, 10%, 54%)"
---

## Overview

PINE is a local desktop web app for interview transcription and analysis, running on Windows, macOS and Linux (Windows is the primary target; the macOS chrome is why the top bar leaves a traffic-light zone). The UI is calm and focused — built for long work sessions where distraction must be minimal. The palette is cool gray (hsl 220) with a green accent (`hsl(152, …)`). Both light and dark themes are first-class: the app ships with a theme toggle and respects user preference.

**Source of truth:** `backend/app/static/css/tokens.css`. Companion sheets: `fonts.css` (bundled `@font-face`), `button-system.css` (every button), `ui-system.css` (everything that is not a button: fields, labels, modal and its option rows, empty state, top bar, toggle switch, colour grid, export options) and `save-status.css`. All five templates link them in that order **before** their own `<style>` block (onboarding has no save status and skips that sheet), so at equal specificity a page's own rule overrides a shared one — which is what a local override should do. Until 2026-11 the shared sheets loaded *after* the page block and silently won. The values in this document mirror `tokens.css` — if the two disagree, the CSS wins.

**Checking it:** `python backend/tools/design_audit.py --rules` lists every declaration that breaks a rule below, as `file:line`, and exits non-zero if there is one. Its exemptions live beside the check, each with its reason. `backend/tests/test_design_audit.py` keeps its spacing scale equal to `tokens.css`.

A colour now changes in one place. The private copies the six templates used to
carry — 168 palette declarations and 117 `@font-face` blocks — are gone; what
remains in a template's `:root` is only what that page alone uses (`main.html`
keeps its layout metrics, `recording.html` its player and tag-label colours).

One rule survived the cleanup and is worth stating: **scope theme selectors to
`:root`.** Four templates wrote a bare `[data-theme="dark"] { … }`, which
matches *any* element carrying the attribute — including the Settings theme
picker's own "Dark" button, which would paint itself in the dark palette while
sitting on a light page. `tokens.css` and every template now use
`:root[data-theme="dark"]`.

The app layout is a two-panel design: a narrow left sidebar for navigation and a main content area for recordings, transcripts, and settings.

## Colors

All color usage goes through semantic CSS custom properties (`var(--token)`), never raw hex or HSL literals in component code.

| Token | Role |
|---|---|
| `--bg` | Page/window background |
| `--fg` | Primary text |
| `--card` | Surface cards, panels, dialogs |
| `--muted` | Subtle backgrounds (hover states, secondary panels) |
| `--muted-fg` | Secondary/placeholder text |
| `--accent` | Primary brand green — buttons, active states, links |
| `--accent-fg` | Text on accent backgrounds (always white) |
| `--accent-dim` | Translucent accent for focus rings, hover tints |
| `--accent-bd` | Accent-colored borders |
| `--border` | Default dividers and input borders |
| `--secondary` | Secondary button backgrounds |
| `--sunken` | Recessed areas (code blocks, input backgrounds) |
| `--destr` | Destructive actions (delete, reset) |
| `--destr-dim` | Translucent destructive for hover tints |
| `--destr-bd` | Destructive-colored borders |
| `--warn` | Warning states |
| `--warn-dim` | Translucent warning for tints |
| `--warn-bd` | Warning-colored borders |
| `--accent-on-dim`, `--destr-on-dim`, `--warn-on-dim` | Contrast-corrected text colour for text sitting **on** the matching `-dim` fill. Never use the solid token for that — it fails contrast in light theme |
| `--accent-bd-solid` | Opaque accent border, where a translucent one would show the layer beneath |

Semantic colors travel in triplets: a solid color (`--accent`), a translucent fill (`--accent-dim`, 10–12% opacity), and a border (`--accent-bd`, 25–28% opacity). Always use all three together when building a new colored interactive surface. Text on the fill takes the fourth token, `-on-dim`.

### Tag categories

Qualitative coding categories carry their own scale, per theme: `--tag-pain`, `--tag-ins` (insight), `--tag-del` (delight), `--tag-conf` (confusion), `--tag-fu` (follow-up), each with `-bg` and `-bd` variants. These are **content** categories, not UI status — keep them separate from `--destr` / `--warn` even where the hues nearly match, so that restyling an alert never restyles a researcher's coding.

## Typography

**Body font** (`--font-body`): Inter, bundled as WOFF2 in `static/fonts/` and declared in `fonts.css`. The value in `tokens.css` is only the pre-JS fallback — each template's boot script overwrites it from the user's `font_family` setting (backend default: `inter`).

**Mono font** (`--font-mono`): JetBrains Mono, bundled as WOFF2 in 400 and 500. It
carries **data, not prose**: timecodes and durations, counts and statistics, file
paths, tokens, versions, and installer output. It is on ~46 rules across the six
screens. Transcript text itself is *not* mono — `.utt-text` deliberately inherits
the body face, and only the timecode beside it is monospaced.

Do not set mono on a sentence. A phrase like "1 tagged quote across 1 recording"
reads as console output when monospaced, even though it is a panel subtitle.

**Font size scale** — six steps, nothing else. The root font size is set by the
user (Settings, default 13px), so every size is a `rem` from this scale — never `px`.

| Step | At 13px root | Role |
|---|---|---|
| `0.769rem` | 10px | uppercase labels (`var(--label-size)`), timestamps, meta, badges |
| `0.846rem` | 11px | small text, sidebar items, secondary descriptions |
| `0.923rem` | 12px | default UI text (most elements) |
| `1rem` | 13px | emphasised text, dialog body copy |
| `1.077rem` | 14px | section titles, dialog titles, editable content and transcript text |
| `1.538rem` | 20px | screen title — one per screen, nothing else |

Buttons never declare `font-size` of their own: the size comes from
`--btn-font-sm/md/lg` in `button-system.css` via the button class. A component
that needs a different button size picks a different button class.

The user can override the body font family and font size at runtime through Settings. Components must use `var(--font-body)` so the override propagates.

**Line height** — four, by job: `1` a single-line control, `1.3` a heading, `1.5` UI text, `1.65` long-form reading (descriptions, editable content, the transcript). Tokens: `--lh-control`, `--lh-heading`, `--lh-ui`, `--lh-prose`.

**Letter spacing** — none by default. Uppercase micro-labels use `--label-tracking` (0.08em); the PINE wordmark uses `--track-wordmark` (0.14em). No negative tracking, and never in `px`.

## Layout

The app shell is a fixed-height window (no scroll on the outer frame). Layout uses CSS flexbox and grid.

- **Left sidebar**: fixed-width, contains navigation and project list
- **Main panel**: fills remaining space, scrolls internally
- **Top bar**: `48px` (`--h-topbar`), defined once in `ui-system.css` and used by every screen but onboarding. `main.html`'s `--top-bar-h: 20px` is not the bar: it is the right header's top padding, which lines the project title up with the first rule in the left column
- **Left header**: `6px` vertical padding, `6px` gap between elements
- Global border-radius for surfaces: `6px` (buttons use `8px`). Radius scale: `--radius-xs` 2px (marks in running text, drag lines, the resize grip) · `--radius-sm` 4px · `--radius` 6px · `--radius-lg` 10px · `--radius-pill` 999px

### Spacing

Twelve steps in three registers — every `padding`, `margin` and `gap` uses one of them:

| Register | Steps | Use |
|---|---|---|
| micro | 2 · 4 · 6 · 8 · 10 | inside controls: icon gaps, chip padding |
| block | 12 · 16 · 20 · 24 · 32 | between elements and inside panels |
| page | 40 · 48 | screen gutters, empty-state padding |

Shared sheets and new code write them as `var(--space-N)`; a template's own `<style>` may keep the literal `px`, but only a value on the scale. Not part of the scale: negative margins and `top/left/right/bottom` (offsets, not rhythm), the two scroll tails below a list (60px, 80px), and three 1px optical nudges — each named in `design_audit.py`. Six and ten stay on the scale on purpose: dropping them would tighten the whole interface.

### Control heights

Buttons and icon buttons take theirs from `button-system.css`. Everything else: `--h-field` 32px, `--h-field-sm` 28px (the search inputs), `--h-topbar` 48px. Apply them as `min-height`, never `height` — the root font size goes up to 17px and a fixed height clips the text.

### Stacking

Never write a raw `z-index` number. The ladder lives in `tokens.css`:

| Token | Value | Use |
|---|---|---|
| `--z-base` | 0 | Default flow |
| `--z-raised` | 1 | Playhead, hover lifts |
| `--z-sticky` | 100 | Sticky headers |
| `--z-dropdown` | 200 | Menus |
| `--z-overlay` | 500 | Full-pane overlays |
| `--z-modal` | 1000 | Modal dialogs and their backdrop |
| `--z-popover` | 1200 | Popovers above a modal |
| `--z-toast` | 2000 | Toast notifications |
| `--z-max` | 9999 | Last resort — a blocking full-screen layer |

A page may collapse adjacent steps — `recording.html`'s popovers are mutually exclusive and all sit on `--z-popover`. `save-status.css` sits at a literal 2100, deliberately above `--z-toast`.

## Components

### Buttons

All interactive controls inherit from a unified base class. Over 40 semantic button classes (`.btn-primary`, `.btn-outline`, `.btn-delete-confirm`, `.seg-chip`, `.rtab`, etc.) share the same geometry tokens.

**Size variants:**

| Size | Height | Padding-X | Font |
|---|---|---|---|
| sm | 30px | 10px | 0.846rem |
| md (default) | 34px | 14px | 0.923rem |
| lg | 38px | 16px | 1rem |

**Icon-only buttons**: use icon-size tokens (`20px` xs / `22px` sm / `28px` md) — width equals height.

**Focus ring**: `box-shadow: var(--btn-focus-ring)` (`0 0 0 3px var(--accent-dim)`) — never use `outline` directly. Every text field gets it on `:focus-visible` from one low-specificity rule in `ui-system.css`, on top of its border colour; a component that draws its own indicator (the segment stepper) turns it off.

**Transition**: all interactive state changes animate at `0.15s ease` across background, border, color, box-shadow, opacity, and transform. Motion that tracks data or a notification — a progress fill, a toast fading — is not interactive and keeps its own duration.

### Elevation and rings

`box-shadow` does three different jobs; keep them apart.

- **Elevation** — only the three tokens: `--shadow-sm` (a card lifted off the page), `--shadow-md` (anything that opens over the page: dropdown, popover, tooltip, toast), `--shadow-lg` (modals). The dark theme has its own alphas. One exception, the back-to-recording button's rest/hover lift pair, waits for a hover-lift token.
- **Rings** — width tokens that compose with any colour: `--ring-hairline` (`0 0 0 1px`) is an outline standing in for a border, as on inline-edit fields; `--ring-focus` (`0 0 0 3px`) is the focus width. A hairline is never a focus indicator.
- **Inset rules** — `inset` shadows used as drop indicators or an edit-mode edge stay local, with a comment.

### Toggle switch, colour grid, dialog rows

`.sw` is 36×20 with a 16px knob, in `ui-system.css`; disabled dims to 0.45 like a disabled button. `.color-grid` is six 22px swatches across with a 6px gap.

Option rows in a dialog (`.m-row`, `.m-lbl`, `.m-sub`) match the rows in Settings that offer the same choices: 10px vertical padding, a hairline between rows, the label in `--fg`. The export dialogs on main and recording and find-and-replace all use them; the format picker and its hint (`.fmt-pick`, `.fmt-hint`) are shared too.

## Do's and Don'ts

**Do:**
- Always use `var(--token)` — never hardcode colors, radii, or font sizes
- Pair semantic colors with their dim + border siblings when building colored states
- Use `var(--font-body)` on all text elements so the user's font preference applies
- Apply focus rings via `box-shadow: var(--btn-focus-ring)` — not browser default outline

**Don't:**
- Don't use `hsl(...)` or `#hex` literals in component CSS — they break theme switching
- Don't add new one-off font sizes outside the six-step scale, and never size text in `px`
- Don't create new button classes without inheriting from the unified base selector list in `button-system.css`
- Don't use `outline` for focus indicators — it conflicts with the macOS window chrome
- Don't write a spacing value off the scale, a raw radius, colour or `z-index`, or a shadow that is not a token — `design_audit.py --rules` will name the line
