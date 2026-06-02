import pandas as pd
import numpy as np
import torch


def get_refractive_index(material: str, wavelength_um: float, csv_path="材料参数.csv") :

    df = pd.read_csv(csv_path).set_index("Type")
    if material == "Air" or material not in df.index:
        return 1.0 if material == "Air" else None

    row = df.loc[material]
    coeffs = row[["K1/A0", "L1/A1", "K2/A2", "L2/A3", "K3/A4", "L3/A5"]].astype(float).values
    lam = wavelength_um
    lam2 = lam * lam

    if row["formula"] == "sellmeier_1":
        K1, L1, K2, L2, K3, L3 = coeffs
        n2 = 1.0 + (K1 * lam2 / (lam2 - L1) + K2 * lam2 / (lam2 - L2) + K3 * lam2 / (lam2 - L3))

    elif row["formula"] == "schott":
        a0, a1, a2, a3, a4, a5 = coeffs
        lam_m2 = 1.0 / lam2
        n2 = (a0 + a1 * lam2 + a2 * lam_m2 + a3 * lam_m2**2 + a4 * lam_m2**3 + a5 * lam_m2**4)
    else:
        raise ValueError(f"未知公式类型: {row['formula']}")

    return np.sqrt(n2)



poly_start_col = None
poly_coeffs = { (0, 2): 1.5, (1, 1): -0.5 }
for (i,b),a in poly_coeffs.items():
    print(i)

