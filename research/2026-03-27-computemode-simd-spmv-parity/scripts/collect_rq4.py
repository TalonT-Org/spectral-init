import json, pathlib

fixtures_dir = pathlib.Path(__file__).parents[3] / "tests" / "fixtures"
results_dir  = pathlib.Path(__file__).parents[1] / "results"
output = []
for meta_path in sorted(fixtures_dir.glob("*/meta.json")):
    meta = json.load(open(meta_path))
    env  = meta.get("env", {})
    output.append({
        "fixture":        meta_path.parent.name,
        "scipy_version":  env.get("scipy"),
        "numpy_version":  env.get("numpy"),
        "python_version": env.get("python"),
        "platform":       env.get("platform"),
    })
json.dump(output, open(results_dir / "scipy_backend.json", "w"), indent=2)
