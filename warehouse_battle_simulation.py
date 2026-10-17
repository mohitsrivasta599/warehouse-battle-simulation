%pip -q install stable-baselines3 gymnasium ipywidgets matplotlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import display, HTML
from IPython.display import clear_output

GRID = 10
MAX_TEAM = 10
MOVES = np.array([
    [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]
])

def clip(p):
    return np.clip(p, 0, GRID - 1).astype(int)

def dist(a, b):
    return abs(int(a[0])-int(b[0])) + abs(int(a[1])-int(b[1]))

class BattleEnv(gym.Env):
    def __init__(self, agents=5, enemies=5, random_counts=False):
        super().__init__()
        self.agents_count = agents
        self.enemies_count = enemies
        self.random_counts = random_counts
        self.max_steps = 60
        self.observation_space = spaces.Box(
            -1, 1, shape=(43,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(MAX_TEAM, 5)
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_counts:
            self.na = int(self.np_random.integers(1, 11))
            self.ne = int(self.np_random.integers(1, 11))
        else:
            self.na = self.agents_count
            self.ne = self.enemies_count

        cells = [(r,c) for r in range(GRID) for c in range(GRID)]
        self.np_random.shuffle(cells)
        self.agents = np.array(cells[:self.na], dtype=int)
        self.enemies = np.array(cells[self.na:self.na+self.ne], dtype=int)
        self.steps = 0
        self.tags = 0
        return self.obs(), {}

    def obs(self):
        x = np.full(43, -1, dtype=np.float32)
        x[:2*self.na] = (self.agents.flatten()/9)*2-1
        x[20:20+2*self.ne] = (self.enemies.flatten()/9)*2-1
        x[40] = self.na/10
        x[41] = self.ne/10
        x[42] = self.steps/self.max_steps
        return x

    def greedy(self, team, opponents):
        actions = []
        for p in team:
            target = min(opponents, key=lambda q: dist(p,q))
            options = [dist(clip(p+m), target) for m in MOVES]
            best = min(options)
            candidates = [i for i,d in enumerate(options) if d == best]
            actions.append(int(self.np_random.choice(candidates)))
        return np.array(actions)

    def step(self, action):
        old_d = sum(
            min(dist(a,e) for e in self.enemies)
            for a in self.agents
        )

        # Friendly agents use selected controller's actions.
        for i in range(self.na):
            self.agents[i] = clip(self.agents[i] + MOVES[int(action[i])])

        # Enemies actively chase the nearest agent.
        enemy_actions = self.greedy(self.enemies, self.agents)
        for i in range(self.ne):
            self.enemies[i] = clip(
                self.enemies[i] + MOVES[enemy_actions[i]]
            )

        # Resolve contact; respawn both units in free cells.
        occupied = {tuple(p) for p in self.agents}
        occupied |= {tuple(p) for p in self.enemies}
        for i in range(self.na):
            for j in range(self.ne):
                if np.array_equal(self.agents[i], self.enemies[j]):
                    self.tags += 1
                    occupied.discard(tuple(self.agents[i]))
                    occupied.discard(tuple(self.enemies[j]))
                    free = [
                        (r,c) for r in range(GRID) for c in range(GRID)
                        if (r,c) not in occupied
                    ]
                    if len(free) >= 2:
                        self.np_random.shuffle(free)
                        self.agents[i] = free[0]
                        self.enemies[j] = free[1]
                        occupied.add(tuple(free[0]))
                        occupied.add(tuple(free[1]))

        new_d = sum(
            min(dist(a,e) for e in self.enemies)
            for a in self.agents
        )
        self.steps += 1
        reward = (old_d-new_d)*0.05
        truncated = self.steps >= self.max_steps
        return self.obs(), float(reward), False, truncated, {"tags": self.tags}


# Train PPO on this battle environment; no Greedy fallback.
print("Training PPO... please wait.")
train_env = BattleEnv(random_counts=True)
model = PPO(
    "MlpPolicy", train_env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=5,
    gamma=0.98,
    seed=42,
    verbose=0
)
model.learn(total_timesteps=12000)
print("PPO training complete.")

# Controls
agent_count = widgets.IntSlider(
    value=5, min=1, max=10, description="Agents",
    continuous_update=False
)
enemy_count = widgets.IntSlider(
    value=5, min=1, max=10, description="Enemies",
    continuous_update=False
)
controller = widgets.Dropdown(
    options=["Random", "Greedy", "PPO"],
    value="Greedy", description="Controller"
)
run_button = widgets.Button(
    description="Run Simulation", button_style="success"
)
output = widgets.Output()

def run_simulation(_):
    with output:
        clear_output(wait=True)
        try:
            env = BattleEnv(agent_count.value, enemy_count.value)
            env.reset()
            history_a = [env.agents.copy()]
            history_e = [env.enemies.copy()]

            for _ in range(50):
                if controller.value == "Random":
                    actions = np.random.randint(0, 5, MAX_TEAM)
                elif controller.value == "Greedy":
                    actions = np.full(MAX_TEAM, 4)
                    actions[:env.na] = env.greedy(
                        env.agents, env.enemies
                    )
                else:
                    actions, _ = model.predict(
                        env.obs(), deterministic=True
                    )
                    actions = np.asarray(actions).reshape(-1)

                _, _, _, done, _ = env.step(actions)
                history_a.append(env.agents.copy())
                history_e.append(env.enemies.copy())
                if done:
                    break

            fig, ax = plt.subplots(figsize=(7,7))
            ax.set_xlim(-0.5, 9.5)
            ax.set_ylim(9.5, -0.5)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.grid(True, alpha=.3)
            ax.set_title(
                f"{controller.value} | Agents: {env.na} | Enemies: {env.ne}"
            )

            for r in range(10):
                for c in range(10):
                    ax.add_patch(Rectangle(
                        (c-.5,r-.5),1,1,
                        fill=False,edgecolor="lightgray",linewidth=.4
                    ))

            a_plot = ax.scatter([], [], s=180, c="blue", label="Agents")
            e_plot = ax.scatter(
                [], [], s=180, c="red", marker="X", label="Enemies"
            )
            counter = ax.text(.02, 1.02, "", transform=ax.transAxes)
            ax.legend(loc="upper right")

            def update(frame):
                a = history_a[frame]
                e = history_e[frame]
                a_plot.set_offsets(np.c_[a[:,1],a[:,0]])
                e_plot.set_offsets(np.c_[e[:,1],e[:,0]])
                counter.set_text(f"Frame: {frame}/{len(history_a)-1}")
                return a_plot, e_plot, counter

            anim = FuncAnimation(
                fig, update, frames=len(history_a),
                interval=350, blit=False, repeat=True
            )
            plt.close(fig)
            display(HTML(anim.to_jshtml()))
            print("Simulation finished. Contact events:", env.tags)

        except Exception as err:
            print("ERROR:", type(err).__name__, str(err))
            raise

run_button.on_click(run_simulation)
display(widgets.VBox([
    widgets.HTML("<h3>Warehouse Battle Simulation</h3>"),
    agent_count, enemy_count, controller, run_button, output
]))
