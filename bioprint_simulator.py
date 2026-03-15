import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import random

size = 10

actions = ["up","down","left","right","print"]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

Q = {}

grid = np.zeros((size,size))

target = np.zeros((size,size))
target[3:7,3:7] = 1

pos = [0,0]

reward_history = []

slide = 0
simulation_running = False

# ---------- AI FUNCTIONS ----------

def get_state():
    return (pos[0],pos[1])

def choose_action(state):

    if state not in Q:
        Q[state] = {a:0 for a in actions}

    if random.random() < epsilon:
        return random.choice(actions)

    return max(Q[state], key=Q[state].get)

def move(action):

    if action == "up":
        pos[0] = max(0,pos[0]-1)

    if action == "down":
        pos[0] = min(size-1,pos[0]+1)

    if action == "left":
        pos[1] = max(0,pos[1]-1)

    if action == "right":
        pos[1] = min(size-1,pos[1]+1)

def compute_reward():

    return -np.sum(np.abs((grid>0)-(target)))

def update_q(state,action,reward,next_state):

    if state not in Q:
        Q[state] = {a:0 for a in actions}

    if next_state not in Q:
        Q[next_state] = {a:0 for a in actions}

    best_next = max(Q[next_state].values())

    Q[state][action] += alpha*(reward + gamma*best_next - Q[state][action])

def step_ai():

    state = get_state()
    action = choose_action(state)

    if action == "print":

        if target[pos[0],pos[1]] == 1:
            grid[pos[0],pos[1]] = 2
        else:
            grid[pos[0],pos[1]] = -1

    else:
        move(action)

    r = compute_reward()
    next_state = get_state()

    update_q(state,action,r,next_state)

    reward_history.append(r)

# ---------- DRAWING FUNCTIONS ----------

def draw_sim():

    ax2.clear()

    display = np.zeros((size,size,3))

    for i in range(size):
        for j in range(size):

            if target[i,j] == 1:
                display[i,j] = [0.5,0.2,0.7]  # purple target

            if grid[i,j] == 2:
                display[i,j] = [0,1,0]  # correct cell (green)

            if grid[i,j] == -1:
                display[i,j] = [1,0,0]  # wrong cell (red)

    display[pos[0],pos[1]] = [1,1,0]  # printer nozzle

    ax2.imshow(display)
    ax2.set_title("AI Bioprinter")

def draw_slide():

    ax1.clear()

    if slide == 0:

        title.set_text("What Is 3D Bioprinting?")

        ax1.text(0.1,0.5,
        "Scientists can print living cells\nlayer by layer to create tissues.",
        fontsize=14)

        ax1.axis("off")

    elif slide == 1:

        title.set_text("Why Print Tumors?")

        ax1.text(0.1,0.5,
        "Researchers print tumor models\nso they can safely test cancer drugs.",
        fontsize=14)

        ax1.axis("off")

    elif slide == 2:

        title.set_text("Target Tumor Structure")

        ax1.imshow(target)

    elif slide == 3:

        title.set_text("Meet the AI Printer")

        ax1.text(0.1,0.5,
        "The yellow square represents\nthe bioprinter nozzle.",
        fontsize=14)

        ax1.axis("off")

    elif slide == 4:

        title.set_text("How the AI Learns")

        ax1.text(0.1,0.5,
        "The AI tries actions and receives\nrewards when the printed tumor\nmatches the target shape.",
        fontsize=14)

        ax1.axis("off")

    plt.draw()

# ---------- BUTTON CONTROLS ----------

def next_slide(event):
    global slide
    slide += 1
    draw_slide()

def prev_slide(event):
    global slide
    slide = max(0, slide-1)
    draw_slide()

def start_simulation(event):
    global simulation_running
    simulation_running = True

# ---------- ANIMATION ----------

def update(frame):

    if simulation_running:

        step_ai()

        ax1.imshow(target)
        ax1.set_title("Target Tumor")

        draw_sim()

        ax3.clear()
        ax3.plot(reward_history)
        ax3.set_title("Reward Score")

# ---------- FIGURE SETUP ----------

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,4))

title = fig.suptitle("",fontsize=18)

# buttons

ax_prev = plt.axes([0.25,0.02,0.12,0.05])
ax_next = plt.axes([0.40,0.02,0.12,0.05])
ax_start = plt.axes([0.60,0.02,0.18,0.05])

btn_prev = Button(ax_prev,"Previous")
btn_next = Button(ax_next,"Next")
btn_start = Button(ax_start,"Start Simulation")

btn_prev.on_clicked(prev_slide)
btn_next.on_clicked(next_slide)
btn_start.on_clicked(start_simulation)

# legend

legend_text = """
Legend
Yellow = Printer
Green = Correct Cell
Red = Incorrect Cell
Purple = Target Tumor
"""

fig.text(0.85,0.3,legend_text)

draw_slide()

ani = FuncAnimation(fig,update,interval=300)

plt.show()