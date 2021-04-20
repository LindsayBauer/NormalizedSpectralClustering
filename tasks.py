from invoke import task

@task
def run(c, k=-1, n=-1, Random=True):
    c.run("python3.8.5 setup.py build_ext --inplace")
    if not Random:
        bool_type = 0
        c.run(f"python3.8 main.py {k} {n} {bool_type}")
    else:
        bool_type = 1
        c.run(f"python3.8 main.py {k} {n} {bool_type}")
    print("Done building")
