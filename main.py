import os
import sys

import nonlinear, system

INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")


def run(module, path):
    sys.argv = [module.__name__ + ".py", os.path.join(INPUTS, path)]
    module.main()


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task in ("all", "2.1"):
        run(nonlinear, "nonlinear_1.txt")
        run(nonlinear, "nonlinear_2.txt")
    if task in ("all", "2.2"):
        run(system, "system.txt")
    if task not in ("all", "2.1", "2.2"):
        print("python main.py [2.1|2.2]")
    print("\n")


if __name__ == "__main__":
    main()
