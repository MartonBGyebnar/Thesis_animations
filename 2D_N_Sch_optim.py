import numpy as np
from scipy.sparse import block_diag, lil_matrix, eye
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import time

start_time = time.time()


def f(alpha, beta, x, y):
    return alpha - x + np.multiply(np.square(x), y)


def g(alpha, beta, x, y):
    return beta - np.multiply(np.square(x), y)


# Parameters of the numerical solution:
Lx: int = 100
x_values = np.linspace(0, Lx, 2*Lx+1)
dx = x_values[1]
Ly: int = 100
y_values = np.linspace(0, Ly, 2*Ly+1)
dy = y_values[1]
T: int = 100
t_values = np.linspace(0, T, 4000*T+1)
dt = t_values[1]

# Parameters of the model:
D1 = 1
D2 = 6
a = 0.2
b = 0.99

# Steady state:
u0 = a + b
v0 = b / u0 ** 2

# Initial conditions:
np.random.seed(42)
u = u0 + np.random.uniform(-0.1, 0.1, size=len(x_values)*len(y_values))
# + np.random.normal(0, 1, size=len(x_values)*len(y_values))
# u[u <= 0] = 0.01
v = v0 + np.random.uniform(-0.1, 0.1, size=len(x_values)*len(y_values))
# + np.random.normal(0, 1, size=len(x_values)*len(y_values))
# v[v <= 0] = 0.01

u_new = np.copy(u)
v_new = np.copy(v)


def laplacian_mtrx(m, k):
    M = np.diag(np.ones(m-1), -1) + np.diag(-4 * np.ones(m), 0) + np.diag(np.ones(m-1), 1)
    M[0, 1] = 2
    M[-1, -2] = 2

    D_main = block_diag([M] * k, format="lil")
    D_upper = lil_matrix((m*k, m*k))
    D_lower = lil_matrix((m*k, m*k))

    # Insert blocks along the diagonals
    D_upper[:m, m:2 * m] = 2 * np.eye(m)  # First block on +1 diagonal gets 2*I
    D_lower[-m:, -2 * m:-m] = 2 * np.eye(m)  # Last block on -1 diagonal gets 2*I

    # Assign remaining blocks in the +1 and -1 diagonals
    D_upper[np.arange(m, (k-1)*m), np.arange(m, (k-1)*m) + m] = 1
    D_lower[np.arange(m, (k-1)*m), np.arange(m, (k-1)*m) - m] = 1

    # Convert to CSR for efficiency
    D_main = D_main.tocsr()
    D_upper = D_upper.tocsr()
    D_lower = D_lower.tocsr()

    # Final sparse matrix
    D_sparse = D_main + D_upper + D_lower
    return D_sparse


# Initialize matrices A1 and A2 used for computing the next time step:
L = laplacian_mtrx(len(y_values), len(x_values))
A1 = D1 * dt / dx ** 2 * L + eye(len(x_values)*len(y_values), format="csr")
A2 = D2 * dt / dx ** 2 * L + eye(len(x_values)*len(y_values), format="csr")

# Set up animation figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# fig.subplots_adjust(wspace=0.02)
img_u = ax1.imshow(u.reshape((len(y_values), len(x_values)), order="F"), cmap="seismic", extent=[0, Lx, 0, Ly],
                   norm=mcolors.TwoSlopeNorm(vcenter=u0, vmin=0, vmax=2.1))
img_v = ax2.imshow(v.reshape((len(y_values), len(x_values)), order="F"), cmap="seismic", extent=[0, Lx, 0, Ly],
                   norm=mcolors.TwoSlopeNorm(vcenter=u0, vmin=0, vmax=2.1))
ax1.set_title("u koncentrációja")
ax2.set_title("v koncentrációja")
# ax1.set_xticks([0, Lx])
# ax2.set_xticks([0, Lx])

cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
cbar = fig.colorbar(img_v, cax=cbar_ax)
# cbar.set_label("Értékek")

snapshot_frames = np.linspace(0, 10, 11) / 10 * (len(t_values) - 1)
frame_skip = 1000
frames = range(0, len(t_values), frame_skip)

i = 0


# Animation function
def update(frame):
    global u, u_new, v, v_new, i

    for _ in range(frame_skip):
        u_new[:] = A1 @ u + dt * f(a, b, u, v)
        v_new[:] = A2 @ v + dt * g(a, b, u, v)
        u[:], v[:] = u_new, v_new

    i += 1
    print("--- %s percent ---" % (((1000 * i) / (len(t_values) - 1)) * 100))

    img_u.set_array(u.reshape((len(y_values), len(x_values)), order="F"))
    img_v.set_array(v.reshape((len(y_values), len(x_values)), order="F"))
    plt.suptitle(f"u és v koncentrációja t={t_values[frame]:.2f}-kor")

    if frame in snapshot_frames:
        fig_snap, (ax1_snap, ax2_snap) = plt.subplots(1, 2, figsize=(10, 5))
        # fig.subplots_adjust(wspace=0.02)
        img_usnap = ax1_snap.imshow(u.reshape((len(y_values), len(x_values)), order="F"), cmap="seismic",
                                    extent=[0, Lx, 0, Ly], norm=mcolors.TwoSlopeNorm(vcenter=u0, vmin=0, vmax=2.1))
        img_vsnap = ax2_snap.imshow(v.reshape((len(y_values), len(x_values)), order="F"), cmap="seismic",
                                    extent=[0, Lx, 0, Ly], norm=mcolors.TwoSlopeNorm(vcenter=u0, vmin=0, vmax=2.1))
        ax1_snap.set_title("u koncentrációja")
        ax2_snap.set_title("v koncentrációja")
        # ax1_snap.set_xticks([0, Lx])
        # ax2_snap.set_xticks([0, Lx])
        cbar_ax_snap = fig_snap.add_axes((0.92, 0.15, 0.02, 0.7))
        cbar_snap = fig_snap.colorbar(img_vsnap, cax=cbar_ax_snap)
        # cbar_snap.set_label("Értékek")

        img_usnap.set_array(u.reshape((len(y_values), len(x_values)), order="F"))
        img_vsnap.set_array(v.reshape((len(y_values), len(x_values)), order="F"))

        fig_snap.savefig(f"labyrinth5_Sch_snapshot_t{t_values[frame]:.0f}.pdf", format='pdf')
        plt.close(fig_snap)

    return img_u, img_v


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
ani.save("labyrinth5_Sch.gif", writer="pillow")

print("--- %s seconds ---" % (time.time() - start_time))
