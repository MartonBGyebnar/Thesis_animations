import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
import time

start_time = time.time()


def f(alpha, beta, x, y):
    return alpha - beta*x + np.divide(np.square(x), y)


def g(alpha, beta, x, y):
    return np.square(x) - y


# Parameters of numerical solution:
L: int = 40
x_values = np.linspace(0, L, 5*L+1)
dx = x_values[1]
T: int = 40
t_values = np.linspace(0, T, 10000*T+1)
dt = t_values[1]

# Parameters of the model:
D1 = 0.7
D2 = 70
a = 0.05
b = 1

# Steady state:
u0 = (a+1)/b
v0 = u0**2

# Initial conditions:
np.random.seed(42)
u = u0 + np.random.uniform(-0.1, 0.1, size=len(x_values))
# + np.random.normal(0, 1, size=len(x_values))
u[u <= 0] = 0.01
v = v0 + np.random.uniform(-0.1, 0.1, size=len(x_values))
# + np.random.normal(0, 1, size=len(x_values))
v[v <= 0] = 0.01

u_new = np.copy(u)
v_new = np.copy(v)


# Numerical solution:
def tridiag_mod(m):
    matrix = np.diag(np.ones(m-1), -1) + np.diag(-2 * np.ones(m), 0) + np.diag(np.ones(m-1), 1)
    matrix[0, 1] = 2
    matrix[-1, -2] = 2
    return matrix


# Initialize matrices A1 and A2:
A1 = D1 * dt / dx ** 2 * tridiag_mod(len(x_values)) + np.eye(len(x_values))
A2 = D2 * dt / dx ** 2 * tridiag_mod(len(x_values)) + np.eye(len(x_values))

# Animáció készítése:
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, 6.1)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_title("u és v koncentrációjának változása az időben")

# Create two line objects for U and V
line_u, = ax.plot(x_values, u, label="u", color="blue", linewidth=3)
line_v, = ax.plot(x_values, v, label="v", color="red", linestyle="dashed", linewidth=3)
ax.legend(["u", "v"], prop={'size': 18})

frame_skip = 1000
frames = range(0, len(t_values), frame_skip)
snapshot_frames = [0, (len(t_values)-1)/4, 2*(len(t_values)-1)/4, 3*(len(t_values)-1)/4, (len(t_values)-1)]

i = 0


# Animation function
def update(frame):
    global u, u_new, v, v_new, i

    i += 1
    print("--- %s percent ---" % (((frame_skip * i) / (len(t_values) - 1)) * 100))

    # Computation:
    for _ in range(frame_skip):
        u_new[:] = A1 @ u + dt * f(a, b, u, v)
        v_new[:] = A2 @ v + dt * g(a, b, u, v)
        u[:] = u_new
        v[:] = v_new

    line_u.set_ydata(u)
    line_v.set_ydata(v)
    ax.set_title(f"u és v koncentrációja t={t_values[frame]:.2f}-kor")

    if frame in snapshot_frames:
        fig_snap, ax_snap = plt.subplots()
        ax_snap.plot(x_values, u, label="u", color="blue", linewidth=3)
        ax_snap.plot(x_values, v, label="v", color="red", linestyle="dashed", linewidth=3)
        ax_snap.set_xlim(0, L)
        ax_snap.set_ylim(0, 6.1)
        ax_snap.yaxis.set_major_locator(MultipleLocator(1))
        ax_snap.legend(["u", "v"], prop={'size': 18})
        # ax_snap.set_xticks([0, 10, 20, 30, 40])
        # ax_snap.set_yticks([0, 1, 2, 3])
        fig_snap.tight_layout()
        fig_snap.savefig(f"Seed_42_snapshot_t{t_values[frame]:.0f}.pdf", format='pdf')
        plt.close(fig_snap)

    return line_u, line_v


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=True)

ani.save("Seed_42.gif", writer="pillow")

print("--- %s seconds ---" % (time.time() - start_time))
