import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')

df = sns.load_dataset('iris')

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

np.random.seed(1337)

obs = df[df.species == 'setosa'].petal_length
mu = obs[0].mean()
std = obs[0].std()
x = np.linspace(0.5, 2.5, 300)
y = norm.pdf(x, mu, std)
line, = ax.plot(x, y)

ax.set_xlim([0.5, 2.5])
ax.set_ylim([0.0, 5.0])
ax.set_title('Gaussian MLE $N = 1$')
ax.set_xlabel('petal_length')
ax.set_ylabel('density')

def update(i):
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    mu = obs[:i].mean()
    std = obs[:i].std()
    # print obs[:i]
    # print mu, std
    line.set_ydata(norm.pdf(x, mu, std))
    ax.set_xlabel('petal_length $\hat{\mu} = %.2f, \hat{\sigma} = %.2f$' % (mu, std))
    ax.set_title('Gaussian MLE $N = {}$'.format(i))
    return line, ax

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(2, 51), interval=500)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('mle.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()