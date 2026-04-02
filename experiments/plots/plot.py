import matplotlib.pyplot as plt


def simple_plot(x: list, y: list, show_window: bool = False, filename: str = "plot.png") -> None:
    plt.plot(x, y)
    plt.savefig(filename)

    if show_window:
        plt.show()

    plt.close()
