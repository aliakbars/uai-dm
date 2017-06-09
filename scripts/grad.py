import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

np.random.seed(1337)

# Plot a scatter that persists (isn't redrawn) and the initial line.
x = np.arange(0, 20, 0.1)
y = x + np.random.normal(0, 3.0, len(x))
ax.scatter(x, y)

eta = 0.001

X = np.column_stack((np.ones(200), x))
w = np.random.randn(2)
line, = ax.plot(x, np.dot(X, w), 'r-', linewidth=2)

ax.set_title('Initialisation')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(i):
    global w
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    # loss = np.dot(X, w) - y
    # grad = loss.dot(X) / len(x)
    # Loss function = \partial E / \partial w
    loss = np.dot(X, w) - y
    grad = np.random.choice(loss)
    w = w - eta * grad
    line.set_ydata(np.dot(X, w))
    ax.set_title('Iteration {}'.format(i))
    return line, ax

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()