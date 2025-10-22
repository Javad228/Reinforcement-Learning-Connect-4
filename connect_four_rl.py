import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox

class ConnectFour:
    def __init__(self):
        self.r = 6
        self.c = 7
        self.grid = np.zeros((self.r, self.c), dtype=int)
        self.turn = 1  # start

    def reset(self):
        self.grid = np.zeros((self.r, self.c), dtype=int)
        self.turn = 1
        return self.get_state()

    def get_state(self):
        return self.grid.tobytes()

    def get_valid_moves(self):
        ok = []
        for j in range(self.c):
            if self.grid[0, j] == 0:
                ok.append(j)
        return ok

    def make_move(self, j):
        if j not in self.get_valid_moves():
            return None, -10, True
        # drop
        rr = None
        for i in range(self.r - 1, -1, -1):
            if self.grid[i, j] == 0:
                self.grid[i, j] = self.turn
                rr = i
                break

        w = self.check_winner()
        done = (w != 0) or (len(self.get_valid_moves()) == 0)
        if w == self.turn:
            rew = 1
        elif w != 0:
            rew = -1
        elif done:
            rew = 0
        else:
            rew = 0
        # swap
        self.turn = 3 - self.turn
        return self.get_state(), rew, done

    def check_winner(self):
        g = self.grid
        # rows
        for i in range(self.r):
            for j in range(self.c - 3):
                x = g[i, j]
                if x != 0 and g[i, j+1] == x and g[i, j+2] == x and g[i, j+3] == x:
                    return x
        # cols
        for i in range(self.r - 3):
            for j in range(self.c):
                x = g[i, j]
                if x != 0 and g[i+1, j] == x and g[i+2, j] == x and g[i+3, j] == x:
                    return x
        # diag \
        for i in range(self.r - 3):
            for j in range(self.c - 3):
                x = g[i, j]
                if x != 0 and g[i+1, j+1] == x and g[i+2, j+2] == x and g[i+3, j+3] == x:
                    return x
        # diag /
        for i in range(self.r - 3):
            for j in range(3, self.c):
                x = g[i, j]
                if x != 0 and g[i+1, j-1] == x and g[i+2, j-2] == x and g[i+3, j-3] == x:
                    return x
        return 0

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.qs = defaultdict(lambda: defaultdict(float))
        self.eps = epsilon
        self.lr = alpha
        self.g = gamma

    def get_action(self, s, moves, training=True):
        # pick
        if training and np.random.random() < self.eps:
            return np.random.choice(moves)
        vals = []
        for m in moves:
            vals.append(self.qs[s][m])
        mx = max(vals)
        best = []
        for m, v in zip(moves, vals):
            if v == mx:
                best.append(m)
        return np.random.choice(best)

    def update(self, s, a, r, s2, moves2):
        if moves2:
            nxt = max(self.qs[s2][m] for m in moves2)
        else:
            nxt = 0
        cur = self.qs[s][a]
        self.qs[s][a] = cur + self.lr * (r + self.g * nxt - cur)

def train(num_games=10000, eval_interval=500):
    a1 = QLearningAgent(epsilon=0.1, alpha=0.5, gamma=0.9)
    a2 = QLearningAgent(epsilon=0.1, alpha=0.5, gamma=0.9)
    hist = []
    wr_vs_rnd = []
    for gcount in range(num_games):
        env = ConnectFour()
        s = env.reset()
        both = {1: a1, 2: a2}
        mem = {1: [], 2: []}
        done = False
        # loop
        while not done:
            p = env.turn
            valid = env.get_valid_moves()
            if not valid:
                break
            act = both[p].get_action(s, valid, training=True)
            mem[p].append((s, act))
            s2, r, done = env.make_move(act)
            s = s2
        w = env.check_winner()
        hist.append(w)
        for p in [1, 2]:
            fin = 1 if w == p else (-1 if w != 0 else 0)
            steps = mem[p]
            n = len(steps)
            for idx, (ss, aa) in enumerate(steps):
                if idx < n - 1:
                    # update
                    both[p].update(ss, aa, 0, steps[idx + 1][0], env.get_valid_moves())
                else:
                    both[p].update(ss, aa, fin, s, [])
        if (gcount + 1) % eval_interval == 0:
            wr = evaluate(a1, num_games=100)
            wr_vs_rnd.append((gcount + 1, wr))
            print(f"Games: {gcount + 1}, Win rate vs random: {wr:.2%}")
    return a1, wr_vs_rnd

def evaluate(agent, num_games=100):
    w = 0
    l = 0
    d = 0
    for _ in range(num_games):
        env = ConnectFour()
        s = env.reset()
        done = False
        cnt = 0
        while not done and cnt < 50:
            valid = env.get_valid_moves()
            if not valid:
                break
            if env.turn == 1:
                a = agent.get_action(s, valid, training=False)
            else:
                a = np.random.choice(valid)
            s, r, done = env.make_move(a)
            cnt += 1
        ww = env.check_winner()
        if ww == 1:
            w += 1
        elif ww == 2:
            l += 1
        else:
            d += 1
    return w / num_games

def plot_performance(wr_list):
    if not wr_list:
        return
    xs = [x for x, _ in wr_list]
    ys = [y for _, y in wr_list]
    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker='o', linewidth=2, markersize=6, label='vs Random Player')
    plt.xlabel('Number of Games')
    plt.ylabel('Win Rate')
    plt.title('Agent vs Random')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('connect_four_performance.png', dpi=150)
    plt.close()

class ConnectFourGUI:
    def __init__(self, agent):
        self.agent = agent
        self.env = ConnectFour()
        self.env.reset()

        self.root = tk.Tk()
        self.root.title("Connect Four - Play as Player 2 (Yellow) vs RL Agent")
        self.root.configure(bg='#2c3e50')

        self.cell = 80
        self.pad = 10
        self.done = False

        t = tk.Label(self.root, text="Connect Four (Player=Yellow)",
                     font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        t.pack(pady=10)

        self.stat = tk.Label(self.root, text="Agent (RED)",
                             font=("Arial", 14), bg='#2c3e50', fg='white')
        self.stat.pack(pady=5)

        w = self.env.c * self.cell + 2 * self.pad
        h = self.env.r * self.cell + 2 * self.pad
        self.cv = tk.Canvas(self.root, width=w, height=h, bg='#3498db', highlightthickness=0)
        self.cv.pack(padx=20, pady=10)

        self.cv.bind('<Button-1>', self.on_click)

        self.draw_board()
        self.root.after(300, self.maybe_agent_move_if_needed)

    def draw_board(self):
        self.cv.delete("all")
        # draw
        for i in range(self.env.r):
            for j in range(self.env.c):
                x1 = j * self.cell + self.pad
                y1 = i * self.cell + self.pad
                x2 = x1 + self.cell
                y2 = y1 + self.cell
                self.cv.create_rectangle(x1, y1, x2, y2, fill='#3498db', outline='#2c3e50', width=2)
                v = self.env.grid[i, j]
                if v != 0:
                    color = '#e74c3c' if v == 1 else '#f1c40f'
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    r = self.cell * 0.4
                    self.cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline='#2c3e50', width=3)

    def on_click(self, evt):
        if self.done:
            return
        if self.env.turn != 2:
            return
        j = (evt.x - self.pad) // self.cell
        if j < 0 or j >= self.env.c:
            return
        valid = self.env.get_valid_moves()
        if j not in valid:
            return
        # click
        s, r, d = self.env.make_move(j)
        self.draw_board()
        if d:
            self.end_game()
            return
        self.root.update()
        self.root.after(300, self.maybe_agent_move_if_needed)

    def maybe_agent_move_if_needed(self):
        if self.done:
            return
        if self.env.turn != 1:
            return
        valid = self.env.get_valid_moves()
        if not valid:
            self.end_game()
            return
        st = self.env.get_state()
        # think
        move = self.agent.get_action(st, valid, training=False)
        _, _, d = self.env.make_move(move)
        self.draw_board()
        if d:
            self.end_game()


    def end_game(self):
        # over
        self.done = True
        w = self.env.check_winner()
        if w == 1:
            msg = "LOST"
            self.stat.config(text=msg, fg='#e74c3c')
        elif w == 2:
            msg = "WON"
            self.stat.config(text=msg, fg='#2ecc71')
        else:
            msg = "DRAW!"
            self.stat.config(text=msg, fg='#95a5a6')
        messagebox.showinfo("Game Over", msg)


    def run(self):
        self.root.mainloop()

def play_against_agent(agent):
    ui = ConnectFourGUI(agent)
    ui.run()

if __name__ == "__main__":
    agent, wr = train(num_games=90000, eval_interval=500)
    print(f"Final eval vs random: {wr[-1][1]:.2%}")
    plot_performance(wr)
    play_against_agent(agent)
