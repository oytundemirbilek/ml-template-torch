from typing import Any
import matplotlib.pyplot as plt


def plot_something(data: Any) -> None:
    plt.plot(data)
    plt.tight_layout()
    plt.show()
