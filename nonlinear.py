from common import *


def newton(f, df, x0, eps, max_iter=100):
    xs = [x0]
    for _ in range(max_iter):
        d = df(xs[-1])
        if abs(d) < EPS:
            raise ValueError("f'(x) = 0, метод Ньютона неприменим")
        xs.append(xs[-1] - f(xs[-1]) / d)
        if abs(xs[-1] - xs[-2]) <= eps:
            return xs
    raise RuntimeError("Метод Ньютона не сошёлся")


def simple_iteration(phi, x0, eps, q, max_iter=100000):
    factor = q / (1 - q) if q < 1 else 1.0 # апостериорная погрешности
    xs = [x0]
    for _ in range(max_iter):
        xs.append(phi(xs[-1]))
        if factor * abs(xs[-1] - xs[-2]) <= eps:
            return xs
    raise RuntimeError("Метод простой итерации не сошёлся")


def load(path):
    p = read_params(path)
    consts = constants(p)
    c = {"text": p["f"], "eps": float(p["eps"]), "x0": float(p["x0"])}
    c["lo"], c["hi"] = vec(p["interval"])
    c["scan"] = vec(p["scan"]) if "scan" in p else [c["lo"], c["hi"]]
    c["f"] = compile_expr(p["f"], ["x"], consts)
    c["phi"] = compile_expr(p["phi"], ["x"], consts)
    c["df"] = compile_expr(p["df"], ["x"], consts) if "df" in p else (lambda x: d1(c["f"], x))
    c["dphi"] = compile_expr(p["dphi"], ["x"], consts) if "dphi" in p else (lambda x: d1(c["phi"], x))
    return c


def sign_changes(f, lo, hi, step=0.1): # Б-К
    n = round((hi - lo) / step)
    pts = [lo + (hi - lo) * i / n for i in range(n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n) if f(pts[i]) * f(pts[i + 1]) <= 0]


def sample(g, lo, hi, n=400):
    return [g(lo + (hi - lo) * i / n) for i in range(n + 1)]


def analyse(c):
    f, df, dphi = c["f"], c["df"], c["dphi"]
    q = max(abs(v) for v in sample(dphi, c["lo"], c["hi"]))
    ratio = max(abs(f(x) * d2(f, x)) / df(x) ** 2 for x in
                [c["lo"] + (c["hi"] - c["lo"]) * i / 200 for i in range(201)])
    xs_it = simple_iteration(c["phi"], c["x0"], c["eps"], q)
    xs_nt = newton(f, df, c["x0"], c["eps"])
    ref = newton(f, df, c["x0"], 1e-14)[-1]
    return q, ratio, xs_it, xs_nt, ref


def print_table(xs, ref, kind):
    errs = [abs(x - ref) for x in xs]
    print("  k          x_k        |x_k-x_k-1|        ошибка   " + ("e_k/e_k-1" if kind == "it" else "e_k/e_k-1^2"))
    for k, x in enumerate(xs):
        step = f"{abs(x - xs[k - 1]):.2e}" if k else "      -"
        ratio = ""
        if k and errs[k - 1] > 1e-9 and errs[k] > 1e-15:
            ratio = f"{errs[k] / errs[k - 1]:.4f}" if kind == "it" else f"{errs[k] / errs[k - 1] ** 2:.4f}"
        print(f"{k:3d}  {x:14.10f}  {step:>14}  {errs[k]:12.2e}   {ratio}")


def main():
    c = load(input_path("nonlinear_1.txt"))
    q, ratio, xs_it, xs_nt, ref = analyse(c)
    print("2.1 Нелинейное уравнение")
    print(f"f(x) = {c['text']}")
    ints = sign_changes(c["f"], *c["scan"])
    print(f"смена знака f на [{c['scan'][0]}, {c['scan'][1]}]: " + ", ".join(f"[{a:.1f}, {b:.1f}]" for a, b in ints))
    print(f"eps = {c['eps']}, отрезок [{c['lo']}, {c['hi']}], x0 = {c['x0']}")
    print(f"q = max|phi'| = {q:.4f}" + ("" if q < 1 else "  (условие сходимости не выполнено)"))
    print(f"max|f*f''|/f'^2 = {ratio:.4f}" + ("" if ratio < 1 else "  (условие Ньютона не выполнено)"))
    print(f"\nпростые итерации: x = {xs_it[-1]:.10f}, итераций: {len(xs_it) - 1}")
    print_table(xs_it, ref, "it")
    print(f"\nНьютон: x = {xs_nt[-1]:.10f}, итераций: {len(xs_nt) - 1}")
    print_table(xs_nt, ref, "nt")
    print(f"\nf(x) в найденной точке: {c['f'](xs_nt[-1]):.2e}")


if __name__ == "__main__":
    main()
