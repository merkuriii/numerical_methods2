import math
import os
import sys

EPS = 1e-12

NAMES = {k: v for k, v in vars(math).items() if not k.startswith("_")}
NAMES.update(ln=math.log, lg=math.log10, tg=math.tan, abs=abs)


def input_path(name):
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", name)


def read_params(path):
    params = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line:
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def vec(s):
    return [float(t) for t in s.replace(",", " ").split()]


def constants(params):
    out = {}
    for k, v in params.items():
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def compile_expr(expr, args, consts):
    code = compile(expr.strip().replace("^", "**"), "<expr>", "eval")
    env = dict(NAMES)
    env.update(consts)

    def fn(*vals):
        env.update(zip(args, vals))
        return eval(code, {"__builtins__": {}}, env)
    return fn


def d1(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def d2(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def jacobian(fs, x, h=1e-6):
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for j in range(n):
        xp, xm = x[:], x[:]
        xp[j] += h
        xm[j] -= h
        for i in range(n):
            J[i][j] = (fs[i](*xp) - fs[i](*xm)) / (2 * h)
    return J


def solve(A, b): # метод Гаусса с выбором главного элемента
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < EPS:
            raise ValueError("Матрица Якоби вырождена")
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= m * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def vec_norm(v):
    return max(abs(t) for t in v)


def mat_norm(A):
    return max(sum(abs(t) for t in row) for row in A)
