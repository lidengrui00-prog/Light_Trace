import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# ---------- 折射率计算 ----------
def get_refractive_index(material: str, wavelength_um: float, csv_path="材料参数.csv") :

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


# ---------- 自由曲面矢高及梯度（包含基底 + 多项式） ----------
def sag_and_grad(x, z, radius, conic,poly_coeffs, norm_radius):
    """
    计算自由曲面矢高 y_sag 及其对 x, z 的偏导数。
    矢高公式:
        y_sag = base(x,z) + poly(x,z)
        base = c*(x^2+z^2) / (1 + sqrt(1 - (1+k)*c^2*(x^2+z^2)))
        poly = Σ A_{p,q} * (x_norm)^p * (z_norm)^q
    其中 c = 1/radius (若 radius=0 则 c=0), x_norm = x / norm_radius, z_norm = z / norm_radius
    """
    # 曲率 c = 1/radius，若 radius 无效则曲率为0
    if radius is not None and not np.isinf(radius) and abs(radius) > 1e-12:
        c = 1.0 / radius
    else:
        c = 0.0
    k = conic if conic is not None else 0.0

    r2 = x*x + z*z
    # 基底矢高及梯度
    if abs(c) < 1e-12:
        # 平面情况
        base = 0.0
        base_grad_x = 0.0
        base_grad_z = 0.0
    else:
        cr2 = c * r2
        sqrt_term = 1.0 - (1.0 + k) * c * cr2
        if sqrt_term < 0:
            # 超出二次曲面定义域，返回较大值避免求交失败
            base = 1e9
            base_grad_x = 0.0
            base_grad_z = 0.0
        else:
            denominator = 1.0 + np.sqrt(sqrt_term)
            base = cr2 / denominator
            # 梯度推导
            dz_dr2 = c / denominator + 1/2 * (c * cr2) / (denominator**2 * np.sqrt(sqrt_term)) * (1.0 + k) * c
            base_grad_x = 2.0 * x * dz_dr2
            base_grad_z = 2.0 * z * dz_dr2

    # 多项式部分（归一化坐标）
    if norm_radius == 0:
        norm_radius = 1.0
    xn = x / norm_radius
    zn = z / norm_radius

    poly = 0.0
    poly_grad_x = 0.0
    poly_grad_z = 0.0

    for (p, q), coeff in poly_coeffs.items():
        if coeff == 0:
            continue
        term = coeff * (xn ** p) * (zn ** q)
        poly += term
        if p > 0:
            poly_grad_x += coeff * p * (xn ** (p-1)) * (zn ** q) / norm_radius
        if q > 0:
            poly_grad_z += coeff * q * (xn ** p) * (zn ** (q-1)) / norm_radius

    sag = base + poly
    grad_x = base_grad_x + poly_grad_x
    grad_z = base_grad_z + poly_grad_z
    return sag, grad_x, grad_z


# ---------- 表面定义 ----------
class Surface:
    def __init__(self, type_name, y_vertex, thickness, material, radius, conic, norm_radius, poly_coeffs):
        self.type = type_name
        self.y_vertex = y_vertex
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
        C = np.array([0.0, self.y_vertex + R, 0.0])
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
        if abs(D[1]) < 1e-12:
            return None
        t = (self.y_vertex - P0[1]) / D[1]
        if t < 1e-8:
            t = 1e-8

        def f(t_val):
            P = P0 + t_val * D
            x, y, z = P[0], P[1], P[2]
            sag, _, _ = sag_and_grad(x, z, self.radius, self.conic,
                                     self.poly_coeffs, self.norm_radius)
            return y - (self.y_vertex + sag)

        def fprime(t_val):
            P = P0 + t_val * D
            x, y, z = P[0], P[1], P[2]
            _, grad_x, grad_z = sag_and_grad(x, z, self.radius, self.conic,
                                             self.poly_coeffs, self.norm_radius)
            return D[1] - (grad_x * D[0] + grad_z * D[2])

        max_iter = 500
        tol = 1e-10
        for _ in range(max_iter):
            ft = f(t)
            if abs(ft) < tol:
                if t > 1e-8:
                    return t
                else:
                    print("Get wrong point")
                    return None
            fpt = fprime(t)
            if abs(fpt) < 1e-12:
                print("wrong fpt")
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
            C = np.array([0.0, self.y_vertex + R, 0.0])
            N = P - C
            N = N / np.linalg.norm(N)
            return N
        elif self.type == 'XY多项式':
            x, y, z = P[0], P[1], P[2]
            _, grad_x, grad_z = sag_and_grad(x, z, self.radius, self.conic,
                                             self.poly_coeffs, self.norm_radius)
            N = np.array([-grad_x, 1.0, -grad_z])
            N = N / np.linalg.norm(N)
            return N
        else:
            return np.array([0.0, 1.0, 0.0])

    def refract(self, I: np.ndarray, N: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        cos_theta1 = -np.dot(I, N)
        if cos_theta1 < 0:
            N = -N
            cos_theta1 = -np.dot(I, N)

        sin_theta1 = np.sqrt(max(0.0, 1.0 - cos_theta1 * cos_theta1))
        ratio = n1 / n2
        sin_theta2 = ratio * sin_theta1

        if sin_theta2 >= 1.0:
            return None

        cos_theta2 = np.sqrt(1.0 - sin_theta2 * sin_theta2)
        R_dir = ratio * I + (ratio * cos_theta1 - cos_theta2) * N
        return R_dir / np.linalg.norm(R_dir)


# ---------- 解析光学系统（适配 CSV） ----------
def load_optical_system(lens_path: str, material_path: str, wavelength_um: float = 0.5876):
    df_lens = pd.read_csv(lens_path, skiprows=1)

    poly_start_col = None
    for i, col in enumerate(df_lens.columns):
        if 'X^0Y^2' in col:
            poly_start_col = i
            break

    surfaces = []
    y_current = 0.0

    for idx, row in df_lens.iterrows():
        type_name = str(row['表面类型']).strip()
        if type_name == '物面':
            continue

        def to_float_or_none(val):
            if pd.isna(val) or val == '':
                return None
            if isinstance(val, str) and val.lower() == 'infinity':
                return None
            return float(val)

        thickness = to_float_or_none(row['厚度'])
        if thickness is None:
            thickness = 0.0
        material = str(row['材料']).strip()
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

        surf = Surface(type_name, y_current, thickness, material,
                       radius, conic, norm_radius, poly_coeffs)
        surfaces.append(surf)

        y_current += thickness

    def n_material(mat):
        return get_refractive_index(mat, wavelength_um, material_path)

    return surfaces, n_material


# ---------- 光线追迹 ----------
def trace_ray(start_point: Tuple[float, float, float],
              start_dir: Tuple[float, float, float],
              surfaces: List[Surface],
              n_func) -> Dict:
    P = np.array(start_point, dtype=float)
    D = np.array(start_dir, dtype=float)
    D = D / np.linalg.norm(D)

    current_n = 1.0
    stop_info = None
    image_point = None

    for i, surf in enumerate(surfaces):
        if surf.type == '像面':
            if abs(D[1]) < 1e-12:
                continue
            t = (surf.y_vertex - P[1]) / D[1]
            if t > 1e-8:
                image_point = P + t * D
            break

        if surf.type == '球面':
            t = surf.intersect_sphere(P, D)
        elif surf.type == 'XY多项式':
            t = surf.intersect_xy_poly(P, D)
        elif surf.type == '光阑':
            if abs(D[1]) < 1e-12:
                continue
            t = (surf.y_vertex - P[1]) / D[1]
        else:
            continue

        if t is None or t <= 1e-8:
            continue

        P_hit = P + t * D

        if surf.type == '光阑':
            radius = np.sqrt(P_hit[0]**2 + P_hit[2]**2)
            stop_info = (P_hit, radius)
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
                print(f"全反射发生在表面 {i} ({surf.type})，追迹终止")
                break
            D = new_dir
            current_n = n2

        P = P_hit + D * 1e-6

    return {
        'stop_point': stop_info[0] if stop_info else None,
        'stop_radius': stop_info[1] if stop_info else None,
        'image_point': image_point
    }


# ---------- 主程序 ----------
if __name__ == "__main__":
    lens_file = "副本附件1_镜头参数.csv"
    material_file = "材料参数.csv"
    wavelength = 0.5876

    surfaces, n_func = load_optical_system(lens_file, material_file, wavelength)
    print(f"共加载 {len(surfaces)} 个表面")

    start = (12.5, 0.0, 0.0)
    direction = (0.0, 1.0, 0.0)
    result = trace_ray(start, direction, surfaces, n_func)

    if result['stop_radius'] is not None:
        print(f"光线在光阑面上的交点: {result['stop_point']}")
        print(f"光阑半径 (到光轴距离): {result['stop_radius']:.6f} mm")
    else:
        print("光线未经过光阑面")

    if result['image_point'] is not None:
        print(f"像面交点: {result['image_point']}")
    else:
        print("光线未到达像面")