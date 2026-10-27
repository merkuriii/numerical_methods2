import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nonlinear
import system
from common import vec_norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")


def path(name):
    return os.path.join(HERE, "inputs", name)


def log_errors(errs):
    return [max(e, 1e-17) for e in errs]


def plot_equation():
    cases = [nonlinear.load(path(n)) for n in ("nonlinear_1.txt", "nonlinear_2.txt")]
    lo, hi = cases[0]["scan"]
    xs = [lo + (hi - lo) * i / 600 for i in range(601)]
    ys = [cases[0]["f"](x) for x in xs]
    lim = max(abs(min(ys)), 5)
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, color="k", lw=0.8)
    for c in cases:
        root = nonlinear.newton(c["f"], c["df"], c["x0"], 1e-12)[-1]
        plt.axvline(root, color="r", ls="--", lw=0.8)
        plt.annotate(f"x = {root:.4f}", (root, 0), textcoords="offset points", xytext=(5, 10))
    plt.ylim(-lim, lim)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.3)
    plt.title(cases[0]["text"] + " = 0")
    plt.savefig(os.path.join(OUT, "2_1_graph.png"), dpi=130, bbox_inches="tight")
    plt.close()
    return cases


def plot_system():
    c = system.load(path("system.txt"))
    lo, hi = c["scan"]
    g = [lo + (hi - lo) * i / 300 for i in range(301)]
    Z1 = [[c["fs"][0](x, y) for x in g] for y in g]
    Z2 = [[c["fs"][1](x, y) for x in g] for y in g]
    sol = system.newton_system(c["fs"], c["J"], c["x0"], 1e-12)[-1]
    plt.figure(figsize=(6, 5))
    plt.contour(g, g, Z1, levels=[0], colors="tab:blue")
    plt.contour(g, g, Z2, levels=[0], colors="tab:orange")
    plt.plot(*sol, "ro")
    plt.annotate(f"({sol[0]:.3f}, {sol[1]:.3f})", sol, textcoords="offset points", xytext=(8, 8))
    plt.plot([], [], color="tab:blue", label=f"f1 = 0: {c['texts'][0]}")
    plt.plot([], [], color="tab:orange", label=f"f2 = 0: {c['texts'][1]}")
    plt.legend(loc="lower right", fontsize=8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT, "2_2_graph.png"), dpi=130, bbox_inches="tight")
    plt.close()
    return c, sol


def plot_convergence(cases, c, sol):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, case, title in zip(axes, cases, ("2.1, корень 1", "2.1, корень 2")):
        q, _, xs_it, xs_nt, ref = nonlinear.analyse(case)
        ax.semilogy(log_errors([abs(x - ref) for x in xs_it]), "o-", label="простые итерации")
        ax.semilogy(log_errors([abs(x - ref) for x in xs_nt]), "s-", label="Ньютон")
        ax.set_title(title)
    q = system.mat_norm(c["Jphi"](c["x0"]))
    it = system.simple_iteration_system(c["phis"], c["x0"], 1e-8, q)
    nt = system.newton_system(c["fs"], c["J"], c["x0"], 1e-8)
    axes[2].semilogy(log_errors([vec_norm([a - b for a, b in zip(x, sol)]) for x in it]), "o-", label="простые итерации")
    axes[2].semilogy(log_errors([vec_norm([a - b for a, b in zip(x, sol)]) for x in nt]), "s-", label="Ньютон")
    axes[2].set_title("2.2, система")
    for ax in axes:
        ax.set_xlabel("номер итерации k")
        ax.set_ylabel("ошибка")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.savefig(os.path.join(OUT, "convergence.png"), dpi=130, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = plot_equation()
    c, sol = plot_system()
    plot_convergence(cases, c, sol)
    print("графики сохранены в plots/")


if __name__ == "__main__":
    main()
