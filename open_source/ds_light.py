import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# ---------- 折射率计算 ----------
def get_refractive_index(material: str, wavelength_um: float, csv_path="材料参数.csv"):
    df = pd.read_csv(csv_path).set_index("Type")
    if material == "air":
        return 1.0

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


def sag_and_grad(x, y, radius, conic, poly_coeffs, norm_radius):
    if radius is not None and not np.isinf(radius) and abs(radius) > 1e-12:
        c = 1.0 / radius
    else:
        c = 0.0
    k = conic if conic is not None else 0.0

    r2 = x * x + y * y
    if abs(c) < 1e-12:
        base = 0.0
        base_grad_x = 0.0
        base_grad_y = 0.0
    else:
        cr2 = c * r2
        sqrt_term = 1.0 - (1.0 + k) * c * cr2
        denominator = 1.0 + np.sqrt(sqrt_term)
        base = cr2 / denominator
        dz_dr2 = c / denominator + 0.5 * (c * cr2) / (denominator**2 * np.sqrt(sqrt_term)) * (1.0 + k) * c
        base_grad_x = 2.0 * x * dz_dr2
        base_grad_y = 2.0 * y * dz_dr2

    if norm_radius == 0:
        norm_radius = 1.0
    xn = x / norm_radius
    yn = y / norm_radius

    poly = 0.0
    poly_grad_x = 0.0
    poly_grad_y = 0.0

    for (p, q), coeff in poly_coeffs.items():
        if coeff == 0:
            continue
        term = coeff * (xn ** p) * (yn ** q)
        poly += term
        if p > 0:
            poly_grad_x += coeff * p * (xn ** (p - 1)) * (yn ** q) / norm_radius
        if q > 0:
            poly_grad_y += coeff * q * (xn ** p) * (yn ** (q - 1)) / norm_radius

    sag = base + poly
    grad_x = base_grad_x + poly_grad_x
    grad_y = base_grad_y + poly_grad_y
    return sag, grad_x, grad_y


# ---------- 表面定义 ----------
class Surface:
    def __init__(self, type_name, z_vertex, thickness, material, radius, conic, norm_radius, poly_coeffs):
        self.type = type_name
        self.z_vertex = z_vertex
        self.thickness = thickness
        self.material = material
        self.radius = radius
        self.conic = conic
        self.norm_radius = norm_radius if norm_radius is not None else 1.0
        self.poly_coeffs = poly_coeffs if poly_coeffs else {}

    def intersect_sphere(self, P0, D):
        if self.radius is None or np.isinf(self.radius):
            return None
        R = self.radius
        C = np.array([0.0, 0.0, self.z_vertex + R])
        delta = P0 - C
        a = np.dot(D, D)
        if abs(a) < 1e-12:
            return None
        b = 2.0 * np.dot(delta, D)
        c = np.dot(delta, delta) - R * R
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t = None
        for ti in (t1, t2):
            if ti > 1e-8:
                if t is None or ti < t:
                    t = ti
        return t

    def intersect_xy_poly(self, P0, D):
        """使用牛顿法求光线与自由曲面的交点"""
        if abs(D[2]) < 1e-12:
            return None
        t = (self.z_vertex - P0[2]) / D[2]
        if t < 1e-8:
            t = 1e-8

        def f(t_val):
            P = P0 + t_val * D
            x, y, z = P[0], P[1], P[2]
            sag, _, _ = sag_and_grad(x, y, self.radius, self.conic,
                                     self.poly_coeffs, self.norm_radius)
            return z - (self.z_vertex + sag)

        def fprime(t_val):
            P = P0 + t_val * D
            x, y, z = P[0], P[1], P[2]
            _, grad_x, grad_y = sag_and_grad(x, y, self.radius, self.conic,
                                             self.poly_coeffs, self.norm_radius)
            return D[2] - (grad_x * D[0] + grad_y * D[1])

        max_iter = 500
        tol = 1e-10
        for _ in range(max_iter):
            ft = f(t)
            if abs(ft) < tol:
                if t > 1e-8:
                    return t
                else:
                    return None
            fpt = fprime(t)
            if abs(fpt) < 1e-12:
                return None
            t_new = t - ft / fpt
            if abs(t_new - t) < tol:
                t = t_new
                break
            t = t_new
        return t

    def get_normal(self, P):
        if self.type == '球面':
            R = self.radius
            C = np.array([0.0, 0.0, self.z_vertex + R])
            N = P - C
            N = N / np.linalg.norm(N)
            return N
        elif self.type == 'XY多项式':
            x, y, z = P[0], P[1], P[2]
            _, grad_x, grad_y = sag_and_grad(x, y, self.radius, self.conic,
                                             self.poly_coeffs, self.norm_radius)
            N = np.array([-grad_x, -grad_y, 1])
            N = N / np.linalg.norm(N)
            return N
        else:
            return np.array([0.0, 0.0, 1.0])

    def refract(self, I: np.ndarray, N: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        cos_theta1 = -np.dot(I, N)
        sin_theta1 = np.sqrt(1.0 - cos_theta1 * cos_theta1)
        ratio = n1 / n2
        sin_theta2 = ratio * sin_theta1

        if sin_theta2 >= 1.0:
            return None

        cos_theta2 = np.sqrt(1.0 - sin_theta2 * sin_theta2)
        R_dir = ratio * I + (ratio * cos_theta1 - cos_theta2) * N
        return R_dir / np.linalg.norm(R_dir)


# ---------- 解析光学系统（适配 CSV） ----------
def load_optical_system(lens_path, material_path, wavelength_um=0.5876):
    df_lens = pd.read_csv(lens_path, skiprows=1)

    poly_start_col = None
    for i, col in enumerate(df_lens.columns):
        if 'X^0Y^2' in col:
            poly_start_col = i
            break

    surfaces = []
    z_current = 0.0

    for idx, row in df_lens.iterrows():
        type_name = str(row['表面类型'])
        if type_name == '物面':
            continue

        def to_float_or_none(val):
            if isinstance(val, str) and val.lower() == 'infinity':
                return None
            return float(val)

        thickness = to_float_or_none(row['厚度'])
        if thickness is None:
            thickness = 0.0
        material = str(row['材料'])
        radius = to_float_or_none(row['半径'])
        conic = to_float_or_none(row['圆锥系数'])
        if conic is None:
            conic = 0.0
        norm_radius = to_float_or_none(row['归一化半径'])
        if norm_radius is None or norm_radius == 0:
            norm_radius = 1.0

        poly_coeffs = {}
        if type_name == 'XY多项式':
            for col in df_lens.columns[poly_start_col:]:
                val = row[col]
                if pd.notna(val) and val != 0:
                    parts = col.split('Y^')
                    x_part = parts[0]
                    y_part = parts[1]
                    a = int(x_part.split('^')[1])
                    b = int(y_part)
                    poly_coeffs[(a, b)] = float(val)

        surf = Surface(type_name, z_current, thickness, material,
                       radius, conic, norm_radius, poly_coeffs)
        surfaces.append(surf)

        z_current += thickness

    def n_material(mat):
        return get_refractive_index(mat, wavelength_um, material_path)

    return surfaces, n_material


# ---------- 光线追迹（原有版本，用于计算光阑半径） ----------
def trace_ray(start_point, start_dir, surfaces, n_func):
    P = np.array(start_point, dtype=float)
    D = np.array(start_dir, dtype=float)
    D = D / np.linalg.norm(D)

    current_n = 1.0
    stop_point = None
    stop_radius = None
    image_point = None

    for i, surf in enumerate(surfaces):
        if surf.type == '像面':
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]
            image_point = P + t * D
            break

        if surf.type == '球面':
            t = surf.intersect_sphere(P, D)
        elif surf.type == 'XY多项式':
            t = surf.intersect_xy_poly(P, D)
        elif surf.type == '光阑':
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]
        else:
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]

        if t is None or t <= 0:
            break

        P_hit = P + t * D

        if surf.type == '光阑':
            stop_point = P_hit
            stop_radius = np.sqrt(P_hit[0]**2 + P_hit[1]**2)
            P = P_hit
            continue

        N = surf.get_normal(P_hit)
        if np.dot(D, N) > 0:
            N = -N

        n1 = current_n
        n2 = n_func(surf.material)

        if surf.type in ('球面', 'XY多项式'):
            new_dir = surf.refract(D, N, n1, n2)
            if new_dir is None:
                break
            D = new_dir
            current_n = n2

        P = P_hit + D * 1e-6

    return {
        'stop_point': stop_point,
        'stop_radius': stop_radius,
        'image_point': image_point
    }


# ---------- 光线追迹（详细版，记录每个表面的交点及出射方向） ----------
def trace_ray_full(start_point, start_dir, surfaces, n_func):
    P = np.array(start_point, dtype=float)
    D = np.array(start_dir, dtype=float)
    D = D / np.linalg.norm(D)

    current_n = 1.0
    surface_hits = []

    for i, surf in enumerate(surfaces):
        if surf.type == '像面':
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]
            if t <= 0:
                continue
            hit_point = P + t * D
            surface_hits.append({
                'index': i,
                'type': surf.type,
                'material': surf.material,
                'hit_point': hit_point.copy(),
                'exit_dir': D.copy()
            })
            break

        if surf.type == '球面':
            t = surf.intersect_sphere(P, D)
        elif surf.type == 'XY多项式':
            t = surf.intersect_xy_poly(P, D)
        elif surf.type == '光阑':
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]
        else:
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]

        if t is None or t <= 0:
            break

        hit_point = P + t * D
        exit_dir = D.copy()

        if surf.type in ('球面', 'XY多项式'):
            N = surf.get_normal(hit_point)
            if np.dot(D, N) > 0:
                N = -N
            n1 = current_n
            n2 = n_func(surf.material)
            new_dir = surf.refract(D, N, n1, n2)
            if new_dir is None:
                surface_hits.append({
                    'index': i,
                    'type': surf.type,
                    'material': surf.material,
                    'hit_point': hit_point.copy(),
                    'exit_dir': None,
                    'total_internal_reflection': True
                })
                break
            exit_dir = new_dir
            current_n = n2

        surface_hits.append({
            'index': i,
            'type': surf.type,
            'material': surf.material,
            'hit_point': hit_point.copy(),
            'exit_dir': exit_dir.copy()
        })

        P = hit_point + exit_dir * 1e-6
        D = exit_dir

    return surface_hits


# ---------- 主程序 ----------
if __name__ == "__main__":
    lens_file = "副本附件1_镜头参数.csv"
    material_file = "材料参数.csv"
    wavelength = 0.5876

    surfaces, n_func = load_optical_system(lens_file, material_file, wavelength)
    print(f"共加载 {len(surfaces)} 个表面\n")

    # 1. 计算光阑半径（通过追迹入瞳边缘光线）
    start = (0.0, 12.5, 0.0)      # 入瞳半高 12.5 mm
    direction = (0.0, 0.0, 1.0)
    result = trace_ray(start, direction, surfaces, n_func)
    if result['stop_radius'] is not None:
        print(f"光阑半径 R = {result['stop_radius']:.6f} mm\n")
    else:
        print("警告：未找到光阑面或光线未通过光阑\n")

    # 2. 详细追迹同一根光线，打印每个表面的交点和方向余弦
    hits = trace_ray_full(start, direction, surfaces, n_func)

    print("=== 光线通过每个表面的记录 ===\n")
    for hit in hits:
        if hit['type'] == '光阑':
            print(f"表面 {hit['index']} ({hit['type']})")
            print(f"  交点: ({hit['hit_point'][0]:.6f}, {hit['hit_point'][1]:.6f}, {hit['hit_point'][2]:.6f})")
            print(f"  出射方向余弦: ({hit['exit_dir'][0]:.8f}, {hit['exit_dir'][1]:.8f}, {hit['exit_dir'][2]:.8f})\n")
        elif hit['type'] == '像面':
            print(f"表面 {hit['index']} ({hit['type']})")
            print(f"  交点: ({hit['hit_point'][0]:.6f}, {hit['hit_point'][1]:.6f}, {hit['hit_point'][2]:.6f})")
            print(f"  光线方向: ({hit['exit_dir'][0]:.8f}, {hit['exit_dir'][1]:.8f}, {hit['exit_dir'][2]:.8f})\n")
        else:
            print(f"表面 {hit['index']} ({hit['type']}) 材料={hit['material']}")
            print(f"  交点: ({hit['hit_point'][0]:.6f}, {hit['hit_point'][1]:.6f}, {hit['hit_point'][2]:.6f})")
            print(f"  折射后方向余弦: ({hit['exit_dir'][0]:.8f}, {hit['exit_dir'][1]:.8f}, {hit['exit_dir'][2]:.8f})\n")