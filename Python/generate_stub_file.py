def extract(line: str, start_token: str, end_token: str) -> str:
    return line.split(start_token)[1].split(end_token)[0]


with open("Python-C-API/imufusion.c") as file:
    lines = file.readlines()

    constants = [extract(l, '"', '"') for l in lines if "PyModule_AddIntConstant" in l]

    functions = [extract(l, ", ", ")") for l in lines if "PyModule_AddFunctions" in l]

    objects = [extract(l, "&", ",") for l in lines if "add_object(module" in l]

constants = "\n".join([f"{c}: int" for c in constants])

functions = "\n\n".join([f"# {f}\n# TODO" for f in functions])

objects = "\n\n".join([f"# {o}\nclass {o.replace('object', '').replace('_', ' ').title().replace(' ', '')}: ...  # TODO" for o in objects])

with open("ximu3-stubs/__init__.pyi", "w") as file:
    file.write(f"""\
{constants}

{functions}

{objects}
""")
