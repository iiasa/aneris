import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-deep")

xmax = 5
linewidth = 5

x = np.linspace(0, xmax)
for factor in [-0.5, -0.25, 0, 0.25, 0.5]:
    y = np.sin(x) + factor * x
    plt.plot(x, y, linewidth=linewidth)

x = np.linspace(-1, 0)
plt.plot(x, np.sin(x), linewidth=linewidth)

plt.xlim(-1, xmax)
plt.axis("off")
plt.tight_layout()
plt.savefig("_static/logo.svg", transparent=True)
