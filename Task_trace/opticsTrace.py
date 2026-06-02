import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch


# ==================== 折射率计算 ====================
def get_refractive_index(material, wavelength_um, csv_path):
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



# ==================== 矢高与梯度 ====================
def sag_and_grad(x, y, radius, conic, poly_coeffs, norm_radius):
    if radius is not None and not np.isinf(radius) and abs(radius) > 1e-12:
        c = 1.0 / radius
    else:
        c = 0.0
    k = conic if conic is not None else 0.0

    r2 = x*x + y*y
    # 基底矢高及梯度
    if abs(c) < 1e-12:
        base = 0.0
        base_grad_x = 0.0
        base_grad_y = 0.0
    else:
        cr2 = c * r2
        sqrt_term = 1.0 - (1.0 + k) * c * cr2
        denominator = 1.0 + np.sqrt(sqrt_term)
        base = cr2 / denominator
        dz_dr2 = c / denominator + 1/2 * (c * cr2) / (denominator**2 * np.sqrt(sqrt_term)) * (1.0 + k) * c
        base_grad_x = 2.0 * x * dz_dr2
        base_grad_y = 2.0 * y * dz_dr2

    # 多项式部分
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
            poly_grad_x += coeff * p * (xn ** (p-1)) * (yn ** q) / norm_radius
        if q > 0:
            poly_grad_y += coeff * q * (xn ** p) * (yn ** (q-1)) / norm_radius

    sag = base + poly
    grad_x = base_grad_x + poly_grad_x
    grad_y = base_grad_y + poly_grad_y
    return sag, grad_x, grad_y

# ==================== 表面类 ====================
class Surface():
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

    def refract(self, I, N, n1, n2):
        cos_theta1 = -np.dot(I, N)
        sin_theta1 = np.sqrt(1.0 - cos_theta1 * cos_theta1)
        ratio = n1 / n2
        sin_theta2 = ratio * sin_theta1

        if sin_theta2 >= 1.0:
            return None

        cos_theta2 = np.sqrt(1.0 - sin_theta2 * sin_theta2)
        R_dir = ratio * I + (ratio * cos_theta1 - cos_theta2) * N
        return R_dir / np.linalg.norm(R_dir)

# ==================== 加载光学系统 ====================
def load_optical_system(lens_path, material_path, wavelength_um):
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




# ==================== 光线追迹 ====================
def trace_ray(start_point, start_dir, surfaces, n_func):
    P = np.array(start_point, dtype=float)
    D = np.array(start_dir, dtype=float)
    D = D / np.linalg.norm(D)

    current_n = 1.0
    stop_info = None
    image_point = None

    intersect_point = []
    direction = []

    for i, surf in enumerate(surfaces):
        if surf.type == '像面':
            if abs(D[2]) < 1e-12:
                continue
            t = (surf.z_vertex - P[2]) / D[2]
            intersect_point.append(P + t * D)
            direction.append(D)
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
            t = None

        if t is None or t <= 0:
            break

        intersect_point.append(P + t * D)
        direction.append(D)
        P_hit = P + t * D

        if surf.type == '光阑':
            radius = np.sqrt(P_hit[0]**2 + P_hit[1]**2)
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
                break
            D = new_dir
            current_n = n2

        P = P_hit + D * 1e-6

    # 确保 stop_info 和 image_point 存在
    if stop_info is None:
        stop_info = (None, None)
    return {
        'the point of each surface': intersect_point,
        'the direction of each surface': direction,
        'stop_point': stop_info[0],
        'stop_radius': stop_info[1],
        'image_point': image_point
    }

# ============================================================
# 坐标下降法 (Coordinate Descent)
# ============================================================
def optimize_coordinate_descent(target_xy, direction, surfaces, n_func, max_iter=15, tol=1e-5):

    # ---- 找光阑面的 z 坐标 ----
    stop_z = None
    for surf in surfaces:
        if surf.type == '光阑':
            stop_z = surf.z_vertex
            break
    if stop_z is None:
        raise ValueError("系统中未找到光阑面")

    dz = direction[2]
    if abs(dz) < 1e-12:
        raise ValueError("光线方向 z 分量太小，无法到达光阑面")
    t_stop = stop_z / dz
    x0 = target_xy[0] - direction[0] * t_stop
    y0 = target_xy[1] - direction[1] * t_stop


    def error_x(x_val):

        result = trace_ray((x_val, y0, 0.0), direction, surfaces, n_func)
        if result['stop_point'] is None:
            return 1e6
        return result['stop_point'][0] - target_xy[0]

    def error_y(y_val):
        
        result = trace_ray((x0, y_val, 0.0), direction, surfaces, n_func)
        if result['stop_point'] is None:
            return 1e6
        return result['stop_point'][1] - target_xy[1]

  
    def secant_step_1d(f, x_curr, h=0.1):
        
        fx = f(x_curr)
        fxh = f(x_curr + h)
        if abs(fxh - fx) < 1e-12:
            return x_curr
        return x_curr - fx * h / (fxh - fx)

    # ---- 交替优化循环 ----
    for it in range(max_iter):
        # 第 1 步：固定 y0，割线法更新 x0
        x0 = secant_step_1d(error_x, x0)

        y0 = secant_step_1d(error_y, y0)


        # 检查收敛
        result = trace_ray((x0, y0, 0.0), direction, surfaces, n_func)


        px, py, _ = result['stop_point']
        error = math.sqrt((px - target_xy[0])**2 + (py - target_xy[1])**2)

        if error < tol:
            return (x0, y0, 0.0), True, error, it + 1

    return (x0, y0, 0.0), False, error, max_iter


# ============================================================
# 六级环采样 (Hexapolar Rings)
# ============================================================
def hexapolar_rings(density=6):
    num_points_per_ring = 6 * np.arange(1, density + 1)

    all_angles = np.hstack([
        np.linspace(0, 2 * np.pi, n, endpoint=False) + 2 * np.pi * i / n
        for i, n in enumerate(num_points_per_ring, 1)
    ])

    radii = np.repeat(np.arange(1, density + 1), num_points_per_ring) / density

    Px = radii * np.cos(all_angles)
    Py = radii * np.sin(all_angles)

    Px = torch.from_numpy(np.append(Px, 0)).reshape(-1, 1)
    Py = torch.from_numpy(np.append(Py, 0)).reshape(-1, 1)
    return Px, Py


# ============================================================
# 点列分析 (Spot Analysis)
# ============================================================
def spot_analysis(ps, option='centroid'):
    """
    计算点列的 RMS 半径、GEO 半径及参考点坐标。

    """
    ps = ps[..., :2]  # 只取 x, y

    if option == 'centroid':
        ref = torch.mean(ps, dim=0)  

    ps_centered = ps - ref[None, ...]          
    dist2 = torch.sum(ps_centered ** 2, dim=1)  
    rms = torch.sqrt(torch.mean(dist2))         
    geo = torch.sqrt(dist2.max())               
    return rms.item(), geo.item(), ref


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    lens_file = r"C:\python_work\光线追迹\Task_trace\Lens_parameter.csv"
    material_file = r"C:\python_work\光线追迹\Task_trace\material_parameter.csv"

    # ---- 三个波长配置: (波长, 颜色, 标记, 图例标签) ----
    wavelengths = [0.656273, 0.4861327, 0.5875618]
    colors     = ['red',   'blue', 'green']
    markers    = ['^',     '+',    's']
    labels     = ['0.656273 μm', '0.4861327 μm', '0.5875618 μm']

    offset_x = 14.217

    # 光线方向：与 z 轴成 8°
    theta = math.radians(8.0)
    direction = np.array([math.sin(theta), 0.0, math.cos(theta)])

    all_image_points = {}   # wavelength -> np.array
    all_results = {}        # wavelength -> (rms, geo, centroid)

    for wavelength in wavelengths:
        print(f"\n{'='*60}")
        print(f"  追踪波长: {wavelength} μm")
        print(f"{'='*60}")

        surfaces, n_func = load_optical_system(lens_file, material_file, wavelength)

        # 获取光阑半径 R
        ref_result = trace_ray((0.0, 12.5, 0.0), (0.0, 0.0, 1.0), surfaces, n_func)
        R = ref_result['stop_radius']
        print(f"光阑半径 R = {R:.4f} mm")

        # 六级环采样
        a, b = hexapolar_rings(density=6)
        pupil_xy = torch.cat([a, b], dim=1)
        pupil_xy = pupil_xy.numpy() * R
        print(f"六级环采样共 {len(pupil_xy)} 个光阑目标点")

        # 逐个优化并追迹
        image_points_xy = []
        success_count = 0
        total = len(pupil_xy)

        for i, target in enumerate(pupil_xy):
            target_tuple = (target[0], target[1])
            start_pt, success, err, iters = optimize_coordinate_descent(
                target_tuple, direction, surfaces, n_func
            )
            if not success:
                continue

            result = trace_ray(start_pt, direction, surfaces, n_func)
            if result['image_point'] is not None:
                image_points_xy.append(list(result['image_point']))
                success_count += 1

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  进度: {i+1}/{total}  (成功 {success_count})")

        print(f"成功追迹: {success_count} / {total} 条光线")

        if len(image_points_xy) > 0:
            image_points_xy = np.array(image_points_xy)
            image_points_xy[:, 0] -= offset_x

            ps_tensor = torch.tensor(image_points_xy)
            rms, geo, centroid = spot_analysis(ps_tensor, option='centroid')

            cx, cy = centroid[0].item(), centroid[1].item()
            all_image_points[wavelength] = image_points_xy
            all_results[wavelength] = (rms, geo, centroid)

            print(f"  质心坐标: ({cx:.4f}, {cy:.4f}) mm")
            print(f"  RMS 半径: {rms:.4f} mm")
            print(f"  GEO 半径: {geo:.4f} mm")

    # ==================== 三波长点列图 ====================
    print("\n\n" + "=" * 60)
    print("  汇总结果")
    print("=" * 60)
    for wavelength in wavelengths:
        if wavelength in all_results:
            rms, geo, centroid = all_results[wavelength]
            cx, cy = centroid[0].item(), centroid[1].item()
            print(f"  {wavelength} μm:")
            print(f"    质心: ({cx:.4f}, {cy:.4f}) mm")
            print(f"    RMS : {rms:.4f} mm,  GEO: {geo:.4f} mm")

    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)

    for wl, color, marker, label in zip(wavelengths, colors, markers, labels):
        pts = all_image_points.get(wl)
        if pts is not None:
            plt.scatter(pts[:, 0], pts[:, 1],
                        color=color, marker=marker, s=12, zorder=5,
                        label=label)

    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Spot Diagram (Hexapolar Rings, 3 wavelengths)")
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
