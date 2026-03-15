import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
matplotlib.rcParams['font.family'] = 'monospace'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import random

# ── THEME ─────────────────────────────────────────────────────────────────────
BG       = "#0d0f14"
PANEL_BG = "#13161e"
ACCENT   = "#00e5c8"
ACCENT2  = "#ff4f8b"
YELLOW   = "#ffe45e"
GREEN    = "#39ff94"
RED      = "#ff3a5c"
TEXT     = "#d4dae8"
DIM      = "#3a3f52"
PURPLE_C = [0.48, 0.19, 1.0]

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(DIM)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.title.set_color(ACCENT)
    ax.title.set_fontsize(8)
    ax.title.set_fontweight("bold")

# ── CONFIG ────────────────────────────────────────────────────────────────────
size = 10

Q_ACTIONS     = ["print", "skip"]
alpha         = 0.4
gamma         = 0.85
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.75

target = np.zeros((size, size))
target[3:7, 3:7] = 1
TOTAL_TARGET = int(np.sum(target))   # 16

SWEEP_PATH = [(r, c) for r in range(size) for c in range(size)]

# ── RUNTIME STATE ─────────────────────────────────────────────────────────────
Q           = {}
episode     = 1
epsilon     = EPSILON_START
best_score  = 0
best_grid   = np.zeros((size, size))

grid      = np.zeros((size, size))
sweep_idx = 0
pos       = list(SWEEP_PATH[0])

episode_accuracies = []
last_action        = ""
last_was_explore   = False
last_q_vals        = {}
last_state         = (0, 0, 0)
last_step_reward   = 0.0
episode_just_reset = False
completed          = False   # True once 16/16 achieved

slide              = 0
simulation_running = False
TOTAL_SLIDES       = 5
MODE               = "slides"   # "slides" | "sim" | "summary"

# ── Q-LEARNING ────────────────────────────────────────────────────────────────
def get_state():
    r, c = pos
    return (r, c, int(target[r, c] == 1))

def ensure_q(state):
    if state not in Q:
        Q[state] = {a: 0.0 for a in Q_ACTIONS}

def choose_action(state):
    ensure_q(state)
    if random.random() < epsilon:
        return random.choice(Q_ACTIONS), True
    return max(Q[state], key=Q[state].get), False

def step_reward(action):
    r, c = pos
    on_target = target[r, c] == 1
    if action == "print":
        return +10.0 if on_target else -10.0
    else:
        return -5.0 if on_target else +2.0

def update_q(state, action, reward, next_state):
    ensure_q(state)
    ensure_q(next_state)
    best_next = max(Q[next_state].values())
    Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

def reset_episode():
    global grid, pos, sweep_idx, epsilon, episode, episode_just_reset
    global best_score, best_grid, completed
    correct = int(np.sum(grid == 2))
    if correct > best_score:
        best_score = correct
        best_grid  = grid.copy()
    if correct >= TOTAL_TARGET:
        completed = True
    episode_accuracies.append(correct / TOTAL_TARGET)
    grid               = np.zeros((size, size))
    sweep_idx          = 0
    pos                = list(SWEEP_PATH[0])
    episode           += 1
    epsilon            = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_just_reset = True

def step_ai():
    global sweep_idx, pos, last_action, last_was_explore
    global last_q_vals, last_state, last_step_reward, episode_just_reset

    episode_just_reset = False
    state            = get_state()
    action, explored = choose_action(state)
    last_state       = state
    last_action      = action
    last_was_explore = explored
    last_q_vals      = dict(Q[state])

    if action == "print":
        grid[pos[0], pos[1]] = 2 if target[pos[0], pos[1]] == 1 else -1

    reward           = step_reward(action)
    last_step_reward = reward

    sweep_idx += 1
    if sweep_idx >= len(SWEEP_PATH):
        reset_episode()
    else:
        pos = list(SWEEP_PATH[sweep_idx])
        next_state = get_state()
        update_q(state, action, reward, next_state)

# ── SLIDE CONTENT ─────────────────────────────────────────────────────────────
SLIDE_CONTENT = [
    ("WHAT IS 3D BIOPRINTING?",
     "Scientists can print living cells\n"
     "layer by layer -- building tissues\n"
     "one microscopic layer at a time."),
    ("WHY PRINT TUMORS?",
     "Researchers print tumor models\n"
     "so they can safely test cancer drugs\n"
     "without risk to real patients."),
    ("TARGET TUMOR STRUCTURE", None),
    ("MEET THE AI PRINTER",
     "The yellow square is the\n"
     "bioprinter nozzle, scanning\n"
     "every cell row by row."),
    ("HOW THE AI LEARNS",
     "The nozzle visits every cell.\n"
     "At each one, Q-learning decides:\n"
     "  PRINT  or  SKIP?\n\n"
     "Rewards shape the decision:\n"
     "  +10  correct print\n"
     "  -10  wrong print\n"
     "   +2  correctly skipped\n"
     "   -5  missed a target cell\n\n"
     "Early episodes: random guesses.\n"
     "By episode 5-10: fills the tumor.\n\n"
     "Press  START  to watch it learn."),
]

SUMMARY_TEXT = (
    "WHAT YOU JUST WITNESSED\n\n"
    "The AI used a HYBRID MODEL:\n\n"
    "  NAVIGATION  (systematic)\n"
    "  The nozzle swept every cell in\n"
    "  order -- no learning needed here.\n"
    "  This guaranteed full coverage.\n\n"
    "  PRINT DECISION  (Q-learning)\n"
    "  At each cell, the AI asked:\n"
    "  should I print or skip?\n"
    "  It learned from rewards until\n"
    "  it always chose correctly.\n\n"
    "  EPSILON DECAY\n"
    "  Early on: random guesses (explore).\n"
    "  Later: trust the Q-table (exploit).\n"
    "  This is how all RL agents learn.\n\n"
    "Real bioprinters use similar AI\n"
    "to decide when to deposit cells\n"
    "with micron-level precision."
)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
CONTENT_TOP    = 0.10
CONTENT_HEIGHT = 0.78
CONTENT_AXES   = []

def remove_content_axes():
    for ax in CONTENT_AXES:
        ax.remove()
    CONTENT_AXES.clear()

def switch_to_slides():
    global ax_slide
    remove_content_axes()
    ax_slide = fig.add_axes([0.03, CONTENT_TOP, 0.94, CONTENT_HEIGHT])
    CONTENT_AXES.append(ax_slide)
    style_ax(ax_slide)
    render_slide_content()
    fig.canvas.draw_idle()

def switch_to_sim():
    global ax_sim, ax_best, ax_reward, ax_commentary
    remove_content_axes()
    # Carefully spaced: sim | best | chart | commentary
    # left margins prevent title text from running into neighbours
    ax_sim        = fig.add_axes([0.03, CONTENT_TOP, 0.17, CONTENT_HEIGHT])
    ax_best       = fig.add_axes([0.22, CONTENT_TOP, 0.17, CONTENT_HEIGHT])
    ax_reward     = fig.add_axes([0.42, CONTENT_TOP, 0.19, CONTENT_HEIGHT])
    ax_commentary = fig.add_axes([0.65, CONTENT_TOP, 0.33, CONTENT_HEIGHT])
    for ax in (ax_sim, ax_best, ax_reward, ax_commentary):
        CONTENT_AXES.append(ax)
        style_ax(ax)
    render_sim()
    render_best()
    render_reward()
    render_commentary()
    fig.canvas.draw_idle()

def switch_to_summary():
    global ax_slide
    remove_content_axes()
    ax_slide = fig.add_axes([0.03, CONTENT_TOP, 0.94, CONTENT_HEIGHT])
    CONTENT_AXES.append(ax_slide)
    style_ax(ax_slide)
    render_summary()
    fig.canvas.draw_idle()

# ── RENDER: SLIDES ────────────────────────────────────────────────────────────
def render_slide_content():
    ax_slide.clear()
    style_ax(ax_slide)
    idx       = min(slide, TOTAL_SLIDES - 1)
    ttl, body = SLIDE_CONTENT[idx]
    ax_slide.text(0.98, 0.97, f"{idx+1} / {TOTAL_SLIDES}",
                  transform=ax_slide.transAxes,
                  ha="right", va="top", fontsize=8, color=DIM)
    if body is None:
        display = np.zeros((size, size, 3))
        for i in range(size):
            for j in range(size):
                if target[i, j] == 1:
                    display[i, j] = PURPLE_C
        ax_slide.imshow(display, interpolation="nearest",
                        extent=[-0.5, size-0.5, size-0.5, -0.5])
        ax_slide.set_xlim(-0.5, size-0.5)
        ax_slide.set_ylim(size+1.5, -2.5)
        ax_slide.set_xticks([])
        ax_slide.set_yticks([])
        ax_slide.text(size/2-0.5, -1.5, ttl,
                      ha="center", va="center",
                      fontsize=13, color=ACCENT, fontweight="bold")
    else:
        ax_slide.axis("off")
        ax_slide.plot([0.1, 0.9], [0.76, 0.76],
                      color=ACCENT, linewidth=1.0, alpha=0.45,
                      transform=ax_slide.transAxes)
        ax_slide.text(0.5, 0.855, ttl,
                      transform=ax_slide.transAxes,
                      ha="center", va="center",
                      fontsize=14, color=ACCENT, fontweight="bold")
        ax_slide.text(0.5, 0.37, body,
                      transform=ax_slide.transAxes,
                      ha="center", va="center",
                      fontsize=10.5, color=TEXT, linespacing=2.0)

# ── RENDER: SUMMARY ───────────────────────────────────────────────────────────
def render_summary():
    ax_slide.clear()
    style_ax(ax_slide)
    ax_slide.axis("off")

    # Two-column layout: left = text, right = final best grid
    # Title
    ax_slide.plot([0.04, 0.96], [0.91, 0.91],
                  color=ACCENT, linewidth=1.0, alpha=0.5,
                  transform=ax_slide.transAxes)
    ax_slide.text(0.5, 0.965, "SIMULATION COMPLETE  --  16/16 ACHIEVED",
                  transform=ax_slide.transAxes,
                  ha="center", va="top",
                  fontsize=13, color=GREEN, fontweight="bold")

    # Left column: explanation text
    # Count printable lines to auto-fit vertical spacing
    lines = SUMMARY_TEXT.split("\n")
    printable = [ln for ln in lines if ln != ""]
    n_lines   = len(printable)
    # Spread from y=0.86 down to y=0.06, leaving clear margin above buttons
    usable    = 0.86 - 0.06
    dy_text   = usable / max(n_lines, 1)
    dy_blank  = dy_text * 0.5

    y = 0.86
    for ln in lines:
        if ln.startswith("WHAT YOU") or ln.startswith("  NAVIGATION") \
                or ln.startswith("  PRINT") or ln.startswith("  EPSILON"):
            col, sz, bold = ACCENT, 9.5, True
        elif ln == "":
            y -= dy_blank
            continue
        else:
            col, sz, bold = TEXT, 9.0, False
        ax_slide.text(0.04, y, ln,
                      transform=ax_slide.transAxes,
                      ha="left", va="top",
                      fontsize=sz, color=col,
                      fontweight="bold" if bold else "normal")
        y -= dy_text

    # Right column: best grid imshow drawn as inset axes
    ax_inset = ax_slide.inset_axes([0.62, 0.10, 0.34, 0.70])
    ax_inset.set_facecolor(PANEL_BG)
    d = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if best_grid[i, j] == 2:
                d[i, j] = [0.22, 1.0, 0.58]
            elif target[i, j] == 1:
                d[i, j] = PURPLE_C
    ax_inset.imshow(d, interpolation="nearest")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for sp in ax_inset.spines.values():
        sp.set_edgecolor(GREEN)
        sp.set_linewidth(1.5)
    ax_slide.text(0.79, 0.84, "FINAL RESULT",
                  transform=ax_slide.transAxes,
                  ha="center", va="bottom",
                  fontsize=9, color=GREEN, fontweight="bold")
    ax_slide.text(0.79, 0.04, f"Achieved in episode {len(episode_accuracies)}",
                  transform=ax_slide.transAxes,
                  ha="center", va="bottom",
                  fontsize=7.5, color=DIM)

# ── RENDER: SIM PANELS ────────────────────────────────────────────────────────
def make_display(g):
    d = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if target[i, j] == 1:
                d[i, j] = PURPLE_C
            if g[i, j] == 2:
                d[i, j] = [0.22, 1.0, 0.58]
            if g[i, j] == -1:
                d[i, j] = [1.0, 0.23, 0.36]
    return d

def render_sim():
    ax_sim.clear()
    style_ax(ax_sim)
    d = make_display(grid)
    if not episode_just_reset and sweep_idx < len(SWEEP_PATH):
        d[pos[0], pos[1]] = [1.0, 0.89, 0.37]
    ax_sim.imshow(d, interpolation="nearest")
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])
    correct = int(np.sum(grid == 2))
    pct     = int(sweep_idx / len(SWEEP_PATH) * 100)
    # Two-line title so nothing overflows
    ax_sim.set_title(f"CURRENT  ep {episode}\n{correct}/{TOTAL_TARGET}  {pct}% done", pad=4)

def render_best():
    ax_best.clear()
    style_ax(ax_best)
    ax_best.imshow(make_display(best_grid), interpolation="nearest")
    ax_best.set_xticks([])
    ax_best.set_yticks([])
    ax_best.set_title(f"BEST SO FAR\n{best_score}/{TOTAL_TARGET} correct", pad=4)

def render_reward():
    ax_reward.clear()
    style_ax(ax_reward)
    if len(episode_accuracies) >= 1:
        xs = list(range(1, len(episode_accuracies)+1))
        ys = [v * 100 for v in episode_accuracies]
        ax_reward.plot(xs, ys, color=ACCENT, linewidth=1.8,
                       marker="o", markersize=3, markerfacecolor=ACCENT)
        ax_reward.fill_between(xs, ys, alpha=0.12, color=ACCENT)
        ax_reward.set_ylim(0, 108)
        ax_reward.axhline(100, color=GREEN, linewidth=0.8,
                          linestyle="--", alpha=0.7)
    ax_reward.set_title("ACCURACY\nper episode", pad=4)
    ax_reward.tick_params(colors=DIM, labelsize=6)
    ax_reward.set_xlabel("episode", color=DIM, fontsize=6, labelpad=1)
    ax_reward.set_ylabel("% correct", color=DIM, fontsize=6, labelpad=1)

def render_commentary():
    ax_commentary.clear()
    style_ax(ax_commentary)
    ax_commentary.axis("off")

    if not last_action:
        ax_commentary.text(0.5, 0.5, "waiting for first step...",
                           transform=ax_commentary.transAxes,
                           ha="center", va="center", color=DIM, fontsize=9)
        return

    y = 0.97
    def line(txt, color=TEXT, size=8.5, bold=False, dy=0.092):
        nonlocal y
        ax_commentary.text(0.04, y, txt,
                           transform=ax_commentary.transAxes,
                           ha="left", va="top",
                           fontsize=size, color=color,
                           fontweight="bold" if bold else "normal")
        y -= dy

    line("Q-LEARNING LOG", color=ACCENT, size=10, bold=True, dy=0.115)

    on_tgt = last_state[2] == 1
    line(f"Cell : row={last_state[0]}, col={last_state[1]}",
         color=DIM, size=8, dy=0.08)
    line("Type : TUMOR -- print!" if on_tgt else "Type : healthy -- skip",
         color=GREEN if on_tgt else TEXT, size=8, dy=0.09)

    y -= 0.008
    line("Q-values (print vs skip):", color=TEXT, size=8.5, bold=True, dy=0.09)
    best_a = max(last_q_vals, key=last_q_vals.get) if last_q_vals else ""
    for a in Q_ACTIONS:
        q_val     = last_q_vals.get(a, 0.0)
        bar_len   = min(10, max(0, int((q_val + 12) / 2.2)))
        bar       = "#" * bar_len
        is_chosen = (a == last_action)
        is_best   = (a == best_a)
        col = ACCENT if is_chosen else (YELLOW if is_best and not last_was_explore else DIM)
        tag = "  <-- chosen" if is_chosen else ""
        line(f"  {a.upper():<6} {q_val:+6.1f}  {bar}{tag}",
             color=col, size=8.5, dy=0.085)

    y -= 0.01
    r = last_step_reward
    if r >= 9:
        rlabel, rcol = f"+{r:.0f}  CORRECT PRINT!", GREEN
    elif r <= -9:
        rlabel, rcol = f"{r:.0f}  WRONG -- penalty!", RED
    elif r > 0:
        rlabel, rcol = f"+{r:.0f}  correct skip", ACCENT
    else:
        rlabel, rcol = f"{r:.0f}  missed target", YELLOW
    line(f"Reward: {rlabel}", color=rcol, size=9, bold=True, dy=0.105)

    y -= 0.008
    if last_was_explore:
        line("MODE: EXPLORE  (random)", color=ACCENT2, size=8.5, bold=True, dy=0.088)
    else:
        line("MODE: EXPLOIT  (best Q)", color=GREEN, size=8.5, bold=True, dy=0.088)
    line(f"  epsilon = {epsilon:.3f}", color=DIM, size=7.5, dy=0.085)

    y -= 0.008
    line(f"Episode : {episode}", color=TEXT, size=8, dy=0.082)
    line(f"Best    : {best_score}/{TOTAL_TARGET} cells correct",
         color=YELLOW, size=8, dy=0.082)

    if episode_just_reset:
        y -= 0.005
        line("-- EPISODE RESET --", color=ACCENT2, size=8.5, bold=True, dy=0.088)

# ── BUTTONS ───────────────────────────────────────────────────────────────────
def style_btn(btn, color=ACCENT):
    btn.color      = PANEL_BG
    btn.hovercolor = color + "22"
    btn.label.set_color(color)
    btn.label.set_fontfamily("monospace")
    btn.label.set_fontsize(9)
    btn.label.set_fontweight("bold")

def on_prev(event):
    global slide, MODE
    if MODE == "summary":
        MODE  = "sim"
        switch_to_sim()
    elif MODE == "sim":
        slide = TOTAL_SLIDES - 1
        MODE  = "slides"
        switch_to_slides()
    elif slide > 0:
        slide -= 1
        render_slide_content()
        fig.canvas.draw_idle()

def on_next(event):
    global slide, MODE
    if MODE == "slides":
        slide += 1
        if slide >= TOTAL_SLIDES:
            MODE = "sim"
            switch_to_sim()
        else:
            render_slide_content()
            fig.canvas.draw_idle()

def on_start(event):
    global simulation_running
    simulation_running = True
    btn_start.label.set_text("RUNNING")
    btn_pause.label.set_text("PAUSE")
    fig.canvas.draw_idle()

def on_pause(event):
    global simulation_running
    simulation_running = not simulation_running
    btn_pause.label.set_text("RESUME" if not simulation_running else "PAUSE")
    fig.canvas.draw_idle()

# ── ANIMATION ────────────────────────────────────────────────────────────────
def update(frame):
    global simulation_running, MODE
    if not simulation_running or MODE != "sim":
        return
    # 1 step per frame = slow enough to follow
    step_ai()
    render_sim()
    render_best()
    render_reward()
    render_commentary()
    # Trigger summary when 16/16 achieved
    if completed and MODE == "sim":
        simulation_running = False
        MODE = "summary"
        switch_to_summary()

# ── FIGURE SETUP ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.5), facecolor=BG)
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.962,
         "AI  BIOPRINTER  //  TUMOR  SIMULATION",
         ha="center", va="top",
         color=ACCENT, fontsize=13, fontweight="bold")

sep = fig.add_axes([0.0, 0.905, 1.0, 0.002])
sep.set_facecolor(DIM)
sep.axis("off")

legend_items = [
    ("* NOZZLE",  YELLOW),
    ("* CORRECT", GREEN),
    ("* WRONG",   RED),
    ("* TARGET",  "#9966ff"),
]
for i, (lbl, col) in enumerate(legend_items):
    fig.text(0.33 + i * 0.057, 0.048, lbl, color=col, fontsize=7.5, va="center")

ax_prev  = fig.add_axes([0.03,  0.015, 0.07, 0.07])
ax_next  = fig.add_axes([0.115, 0.015, 0.07, 0.07])
ax_start = fig.add_axes([0.72,  0.015, 0.11, 0.07])
ax_pause = fig.add_axes([0.845, 0.015, 0.11, 0.07])

btn_prev  = Button(ax_prev,  "< PREV")
btn_next  = Button(ax_next,  "NEXT >")
btn_start = Button(ax_start, "START")
btn_pause = Button(ax_pause, "PAUSE")

for btn, col in [(btn_prev, ACCENT), (btn_next, ACCENT),
                 (btn_start, ACCENT2), (btn_pause, YELLOW)]:
    style_btn(btn, col)
for ax_b in (ax_prev, ax_next, ax_start, ax_pause):
    ax_b.set_facecolor(PANEL_BG)
    for sp in ax_b.spines.values():
        sp.set_edgecolor(DIM)

btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_start.on_clicked(on_start)
btn_pause.on_clicked(on_pause)

ax_slide = ax_sim = ax_best = ax_reward = ax_commentary = None
switch_to_slides()

ani = FuncAnimation(fig, update, interval=120, cache_frame_data=False)
plt.show()