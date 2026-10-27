from common import *


def fmt(v, prec=8):
    return "[" + ", ".join(f"{t:.{prec}f}" for t in v) + "]"


def newton_system(fs, J, x0, eps, max_iter=100):
    xs = [x0[:]]
    for _ in range(max_iter):
        x = xs[-1]
        dx = solve(J(x), [-f(*x) for f in fs])
        xs.append([a + b for a, b in zip(x, dx)])
        if vec_norm(dx) <= eps:
            return xs
    raise RuntimeError("Метод Ньютона не сошёлся")


def simple_iteration_system(phis, x0, eps, q, max_iter=100000):
    factor = q / (1 - q) if q < 1 else 1.0
    xs = [x0[:]]
    for _ in range(max_iter):
        x = xs[-1]
        xs.append([p(*x) for p in phis])
        if factor * vec_norm([a - b for a, b in zip(xs[-1], x)]) <= eps:
            return xs
    raise RuntimeError("Метод простой итерации не сошёлся")


def load(path):
    p = read_params(path)
    consts = constants(p)
    n = sum(1 for k in p if k.startswith("f") and k[1:].isdigit())
    names = [f"x{i + 1}" for i in range(n)]
    c = {"n": n, "names": names, "eps": float(p["eps"]), "x0": vec(p["x0"]), "p": p}
    c["texts"] = [p[f"f{i + 1}"] for i in range(n)]
    c["fs"] = [compile_expr(t, names, consts) for t in c["texts"]]
    c["phis"] = [compile_expr(p[f"phi{i + 1}"], names, consts) for i in range(n)]
    if all(f"J{i + 1}" in p for i in range(n)):
        rows = [[compile_expr(e, names, consts) for e in p[f"J{i + 1}"].split(";")] for i in range(n)]
        c["J"] = lambda x: [[g(*x) for g in row] for row in rows]
    else:
        c["J"] = lambda x: jacobian(c["fs"], x)
    c["Jphi"] = lambda x: jacobian(c["phis"], x)
    c["scan"] = vec(p["scan"]) if "scan" in p else None
    return c


def scan_cells(fs, lo, hi, step=0.25):
    m = round((hi - lo) / step)
    pts = [lo + step * i for i in range(m + 1)]
    cells = []
    for i in range(m):
        for j in range(m):
            corners = [(pts[i + a], pts[j + b]) for a in (0, 1) for b in (0, 1)]
            if all(min(f(*q) for q in corners) <= 0 <= max(f(*q) for q in corners) for f in fs):
                cells.append((pts[i], pts[j]))
    return cells


def print_table(xs, ref):
    print("  k  x                                  |dx|        ошибка")
    shown = range(len(xs)) if len(xs) <= 14 else list(range(10)) + [len(xs) - 1]
    for k in shown:
        x = xs[k]
        if k == len(xs) - 1 and len(xs) > 14:
            print("  ...")
        step = f"{vec_norm([a - b for a, b in zip(x, xs[k - 1])]):.2e}" if k else "      -"
        err = vec_norm([a - b for a, b in zip(x, ref)])
        print(f"{k:3d}  {fmt(x, 8):34}  {step:>9}  {err:10.2e}")


def main():
    c = load(input_path("system.txt"))
    n, eps = c["n"], c["eps"]
    print("2.2 Система нелинейных уравнений")
    for i, t in enumerate(c["texts"]):
        print(f"f{i + 1}: {t} = 0")
    if c["scan"] and n == 2:
        cells = scan_cells(c["fs"], *c["scan"])
        print("клетки сетки с решением (левый нижний угол, шаг 0.25): " + ", ".join(f"({a:.2f}, {b:.2f})" for a, b in cells))
    x0 = c["x0"]
    q = mat_norm(c["Jphi"](x0))
    print(f"eps = {eps}, x0 = {fmt(x0, 4)}")
    print(f"||J_phi(x0)|| = {q:.4f}" + ("" if q < 1 else "  (условие сходимости не выполнено)"))
    xs_it = simple_iteration_system(c["phis"], x0, eps, q)
    xs_nt = newton_system(c["fs"], c["J"], x0, eps)
    ref = newton_system(c["fs"], c["J"], x0, 1e-14)[-1]
    print(f"||J_phi(x*)|| = {mat_norm(c['Jphi'](ref)):.4f}")

    print(f"\nпростые итерации: x = {fmt(xs_it[-1])}, итераций: {len(xs_it) - 1}")
    print_table(xs_it, ref)
    print(f"\nНьютон: x = {fmt(xs_nt[-1])}, итераций: {len(xs_nt) - 1}")
    print_table(xs_nt, ref)
    res = [f(*xs_nt[-1]) for f in c["fs"]]
    print(f"\nневязка f(x) = {fmt(res, 2)}")
    print("все компоненты положительны" if all(t > 0 for t in ref) else "решение не положительное")


if __name__ == "__main__":
    main()
