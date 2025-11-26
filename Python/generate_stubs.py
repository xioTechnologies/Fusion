def extract(line: str, start_token: str, end_token: str) -> str:
    return line.split(start_token)[1].split(end_token)[0]


with open("imufusion/imufusion.c") as file:
    lines = file.readlines()

    constants = [extract(l, '"', '"') for l in lines if "PyModule_AddIntConstant" in l]

    functions = [extract(l, ", ", ")") for l in lines if "PyModule_AddFunctions" in l]

    objects = [extract(l, "&", ",") for l in lines if "add_object(module" in l]

with open("imufusion-stubs/__init__.pyi", "w") as file:
    file.write(f"""\
{"\n".join([f"{c}: int" for c in constants])}

{"\n\n".join([f"# {f}\n# TODO" for f in functions])}

{"\n\n".join([f"# {o}\nclass {o.replace('object', '').replace('_', ' ').title().replace(' ', '')}: ...  # TODO" for o in objects])}
""")
