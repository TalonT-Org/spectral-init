import numpy as np
from scipy.linalg import subspace_angles
import json, pathlib

results_dir = pathlib.Path(__file__).parents[1] / "results"
ev_dir = results_dir / "eigenvectors"

output = []
for fixture in ["blobs_connected_2000", "blobs_5000"]:
    scalar_path = ev_dir / f"{fixture}_scalar.npy"
    avx2_path   = ev_dir / f"{fixture}_avx2.npy"
    if not scalar_path.exists():
        continue
    V_s = np.load(str(scalar_path))  # shape (n, 2)
    V_a = np.load(str(avx2_path))
    angles = subspace_angles(V_s, V_a)  # principal angles in radians
    output.append({
        "fixture": fixture,
        "principal_angles_rad": angles.tolist(),
        "max_angle_rad": float(angles.max()),
    })

json.dump(output, open(results_dir / "subspace_angles.json", "w"), indent=2)
print(json.dumps(output, indent=2))
