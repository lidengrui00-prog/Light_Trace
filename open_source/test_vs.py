"""
光线追迹 —— 支持球面与XY多项式自由曲面
从CSV文件加载光学系统，追迹光线并绘制光路图与点列图。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================== 折射率计算 ====================
def get_refractive_index(material: str, wavelength_um: float, csv_path="材料参数.csv") -> float:
    df = pd.read_csv(csv_path).set_index("Type")
    if material.lower() == "air":
        return 1.0
    row = df.loc[material]
    coeffs = row[["K1/A0", "L1/A1", "K2/A2", "L2/A3", "K3/A4", "L3/A5"]].astype(float).values
    lam2 = wavelength_um ** 2

    if row["formula"] == "sellmeier_1":
        K1, L1, K2, L2, K3, L3 = coeffs
        n2 = 1.0 + K1 * lam2 / (lam2 - L1) + K2 * lam2 / (lam2 - L2) + K3 * lam2 / (lam2 - L3)
    elif row["formula"] == "schott":
        a0, a1, a2, a3, a4, a5 = coeffs
        lam_m2 = 1.0 / lam2
        n2 = a0 + a1 * lam2 + a2 * lam_m2 + a3 * lam_m2 ** 2 + a4 * lam_m2 ** 3 + a5 * lam_m2 ** 4
    else:
        raise ValueError(f"未知公式类型: {row['formula']}")
    return np.sqrt(n2)


# ==================== 矢高与梯度 ====================
def sag_and_grad(x, y, radius, conic, poly_coeffs, norm_radius):
    """计算表面矢高 z(x,y) 及其梯度 (dz/dx, dz/dy)"""
    c = 1.0 / radius if (radius is not None and abs(radius) > 1e-12) else 0.0
    k = conic if conic is not None else 0.0
    r2 = x * x + y * y

    if abs(c) < 1e-12:
        base, base_grad_x, base_grad_y = 0.0, 0.0, 0.0
    else:
        cr2 = c * r2
        sqrt_term = np.sqrt(1.0 - (1.0 + k) * c * cr2)
        denom = 1.0 + sqrt_term
        base = cr2 / denom
        dz_dr2 = (c / denom
                  + 0.5 * c * cr2 / (denom * denom * sqrt_term) * (1.0 + k) * c)
        base_grad_x = 2.0 * x * dz_dr2
        base_grad_y = 2.0 * y * dz_dr2

    # 多项式部分
    if norm_radius == 0:
        norm_radius = 1.0
    xn, yn = x / norm_radius, y / norm_radius
    poly, poly_grad_x, poly_grad_y = 0.0, 0.0, 0.0
    for (p, q), coeff in poly_coeffs.items():
        if coeff == 0:
            continue
        poly += coeff * (xn ** p) * (yn ** q)
        if p > 0:
            poly_grad_x += coeff * p * (xn ** (p - 1)) * (yn ** q) / norm_radius
        if q > 0:
            poly_grad_y += coeff * q * (xn ** p) * (yn ** (q - 1)) / norm_radius

    return base + poly, base_grad_x + poly_grad_x, base_grad_y + poly_grad_y


# ==================== 表面类 ====================
class Surface:
    def __init__(self, type_name, z_vertex, thickness, material,
                 radius, conic, norm_radius, poly_coeffs):
        self.type = type_name
        self.z_vertex = z_vertex
        self.thickness = thickness
        self.material = material
        self.radius = radius
        self.conic = conic
        self.norm_radius = norm_radius if norm_radius is not None else 1.0
        self.poly_coeffs = poly_coeffs if poly_coeffs else {}

    def intersect(self, P0, D):
        """计算光线与表面的交点参数 t"""
        if self.type == "球面":
            return self._intersect_sphere(P0, D)
        elif self.type == "XY多项式":
            return self._intersect_xy_poly(P0, D)
        else:  # 平面（光阑等）
            if abs(D[2]) < 1e-12:
                return None
            t = (self.z_vertex - P0[2]) / D[2]
            return t if t > 1e-8 else None

    def _intersect_sphere(self, P0, D):
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
        t1, t2 = (-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)
        t = None
        for ti in (t1, t2):
            if ti > 1e-8 and (t is None or ti < t):
                t = ti
        return t

    def _intersect_xy_poly(self, P0, D):
        """牛顿法求光线与自由曲面的交点"""
        if abs(D[2]) < 1e-12:
            return None
        t = (self.z_vertex - P0[2]) / D[2]
        if t < 1e-8:
            t = 1e-8

        for _ in range(200):
            P = P0 + t * D
            x, y = P[0], P[1]
            sag, gx, gy = sag_and_grad(x, y, self.radius, self.conic,
                                       self.poly_coeffs, self.norm_radius)
            f = P[2] - (self.z_vertex + sag)
            fp = D[2] - (gx * D[0] + gy * D[1])
            if abs(f) < 1e-10:
                return t if t > 1e-8 else None
            if abs(fp) < 1e-14:
                return None
            t_new = t - f / fp
            if abs(t_new - t) < 1e-10:
                return t_new if t_new > 1e-8 else None
            t = t_new
        return None

    def get_normal(self, P):
        """返回表面在点P处的单位法向量（指向入射侧）"""
        if self.type == "球面":
            R = self.radius
            C = np.array([0.0, 0.0, self.z_vertex + R])
            N = P - C
        elif self.type == "XY多项式":
            x, y = P[0], P[1]
            _, gx, gy = sag_and_grad(x, y, self.radius, self.conic,
                                     self.poly_coeffs, self.norm_radius)
            N = np.array([-gx, -gy, 1.0])
        else:
            N = np.array([0.0, 0.0, 1.0])
        return N / np.linalg.norm(N)

    @staticmethod
    def refract(I, N, n1, n2):
        """Snell折射定律，返回折射方向；全反射返回None"""
        cos1 = -np.dot(I, N)
        sin1_sq = max(0.0, 1.0 - cos1 * cos1)
        ratio = n1 / n2
        sin2_sq = ratio * ratio * sin1_sq
        if sin2_sq >= 1.0:
            return None
        cos2 = np.sqrt(1.0 - sin2_sq)
        R = ratio * I + (ratio * cos1 - cos2) * N
        return R / np.linalg.norm(R)


# ==================== 光学系统加载 ====================
def load_optical_system(lens_path, material_path, wavelength_um=0.5876):
    df = pd.read_csv(lens_path, skiprows=1)

    poly_start_col = None
    for i, col in enumerate(df.columns):
        if "X^0Y^2" in col:
            poly_start_col = i
            break

    surfaces = []
    z_current = 0.0

    for _, row in df.iterrows():
        type_name = str(row["表面类型"])
        if type_name == "物面":
            continue

        def to_float(v):
            if isinstance(v, str) and v.lower() == "infinity":
                return None
            return float(v)

        thickness = to_float(row["厚度"]) or 0.0
        material = str(row["材料"])
        radius = to_float(row["半径"])
        conic = to_float(row["圆锥系数"]) or 0.0
        norm_radius = to_float(row["归一化半径"]) or 1.0

        poly_coeffs = {}
        if type_name == "XY多项式" and poly_start_col is not None:
            for col in df.columns[poly_start_col:]:
                val = row[col]
                if pd.notna(val) and float(val) != 0:
                    parts = col.split("Y^")
                    a = int(parts[0].split("^")[1])
                    b = int(parts[1])
                    poly_coeffs[(a, b)] = float(val)

        surfaces.append(Surface(type_name, z_current, thickness, material,
                                radius, conic, norm_radius, poly_coeffs))
        z_current += thickness

    def n_func(mat):
        return get_refractive_index(mat, wavelength_um, material_path)

    return surfaces, n_func


# ==================== 光线追迹 ====================
def trace_ray(start_point, start_dir, surfaces, n_func):
    """追迹单根光线，返回各面交点、方向及像面/光阑信息"""
    P = np.array(start_point, dtype=float)
    D = np.array(start_dir, dtype=float)
    D = D / np.linalg.norm(D)
    current_n = 1.0

    points, dirs = [], []
    stop_info = None
    image_point = None

    for surf in surfaces:
        if surf.type == "像面":
            if abs(D[2]) < 1e-12:
                break
            t = (surf.z_vertex - P[2]) / D[2]
            pt = P + t * D
            points.append(pt)
            dirs.append(D)
            image_point = pt
            break

        t = surf.intersect(P, D)
        if t is None or t <= 0:
            break

        P_hit = P + t * D
        points.append(P_hit)
        dirs.append(D)

        if surf.type == "光阑":
            stop_info = (P_hit, np.hypot(P_hit[0], P_hit[1]))
            P = P_hit + D * 1e-6
            continue

        N = surf.get_normal(P_hit)
        if np.dot(D, N) > 0:
            N = -N

        n1 = current_n
        n2 = n_func(surf.material)
        new_dir = surf.refract(D, N, n1, n2)
        if new_dir is None:
            break
        D = new_dir
        current_n = n2
        P = P_hit + D * 1e-6

    if stop_info is None:
        stop_info = (None, None)
    return {
        "points": points,
        "dirs": dirs,
        "stop_point": stop_info[0],
        "stop_radius": stop_info[1],
        "image_point": image_point,
    }


# ==================== 批量追迹 & 可视化 ====================
def trace_grid(surfaces, n_func, view_angle_deg=0.0, N=11, R_pupil=12.5):
    """对N×N网格的平行光进行追迹，返回所有像面交点"""
    theta = np.radians(view_angle_deg)
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    direction = direction / np.linalg.norm(direction)

    xs = np.linspace(-R_pupil, R_pupil, N)
    ys = np.linspace(-R_pupil, R_pupil, N)
    image_pts = []

    for x in xs:
        for y in ys:
            result = trace_ray((x, y, 0.0), direction, surfaces, n_func)
            if result["image_point"] is not None:
                image_pts.append(result["image_point"])

    return np.array(image_pts), xs, ys


def _surface_sag_profile(surf, r_vals):
    """计算表面在给定高度数组处的 z 坐标（z_vertex + sag）"""
    sag_vals = np.array([
        sag_and_grad(r, 0.0, surf.radius, surf.conic,
                     surf.poly_coeffs, surf.norm_radius)[0]
        for r in r_vals
    ])
    return surf.z_vertex + sag_vals


def _build_lens_groups(surfaces):
    """
    将连续的非空气表面归为一个镜片组。
    返回: [(front_surf_idx, back_surf_idx, material_list, aperture_radius), ...]
    其中 back_surf_idx 是该镜片最后一个玻璃表面后的那个空气面。
    """
    groups = []
    i = 0
    while i < len(surfaces):
        if surfaces[i].material.lower() == "air" or surfaces[i].type == "像面":
            i += 1
            continue
        # 镜片开始：找连续的非空气表面
        j = i
        while j < len(surfaces) and surfaces[j].material.lower() != "air" \
                and surfaces[j].type != "像面":
            j += 1
        # j 现在是第一个 air 表面（镜片后表面），或者越界
        if j < len(surfaces):
            # 镜片: surfaces[i:j] 是玻璃面, surfaces[j] 是后空气面
            mats = [surfaces[k].material for k in range(i, j + 1)]
            # 取最小通光孔径（半径取绝对值；对于XY多项式用归一化半径）
            r_list = []
            for k in range(i, j + 1):
                r_val = surfaces[k].radius
                if r_val is not None and not np.isinf(r_val):
                    r_list.append(abs(r_val))
                if surfaces[k].type == "XY多项式":
                    r_list.append(surfaces[k].norm_radius)
            r_ap = min(r_list) if r_list else 12.5
            groups.append((i, j, mats, r_ap))
        i = j + 1 if j < len(surfaces) else j
    return groups


def plot_lens_layout(surfaces, n_func, views=(0, 5, 10), R=12.5, M=7,
                     figsize=(14, 7), savepath=None):
    """
    绘制完整光学系统布局图：填充镜片 + 光阑 + 光线追迹。

    参数:
        surfaces: 表面列表
        n_func:  折射率函数
        views:   视场角列表 [度]
        R:       入瞳半径 [mm]
        M:       每个视场的光线数（在入瞳上均匀采样）
        figsize: 图像尺寸
        savepath: 保存路径 (可选)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ---- 1. 绘制填充镜片 ----
    lens_groups = _build_lens_groups(surfaces)

    # 为不同镜片分配颜色（浅蓝/浅绿交替）
    glass_colors = ["#b3d9ff", "#b3e6cc", "#c2c2f0", "#ffe0b3",
                    "#d1c4e9", "#b2dfdb", "#f8bbd0", "#fff9c4"]

    for g_idx, (front_idx, back_air_idx, mats, r_ap) in enumerate(lens_groups):
        color = glass_colors[g_idx % len(glass_colors)]
        front_surf = surfaces[front_idx]
        back_air_surf = surfaces[back_air_idx]

        # 采样前表面和后空气面的轮廓
        rr = np.linspace(-r_ap, r_ap, 300)
        z_front = _surface_sag_profile(front_surf, rr)
        z_back = _surface_sag_profile(back_air_surf, rr)

        # 如果有中间表面（胶合镜），也画出来
        middle_surfs = []
        for mi in range(front_idx + 1, back_air_idx):
            if surfaces[mi].material.lower() != "air":
                middle_surfs.append(mi)

        # 填充镜片体
        ax.fill_betweenx(rr, z_front, z_back, color=color, alpha=0.6, linewidth=0)

        # 画前表面轮廓
        ax.plot(z_front, rr, "k-", linewidth=1.0)
        # 画后表面轮廓
        ax.plot(z_back, rr, "k-", linewidth=1.0)
        # 画边缘线
        for r_edge in [-r_ap, r_ap]:
            ax.plot([z_front[np.argmin(np.abs(rr - r_edge))],
                     z_back[np.argmin(np.abs(rr - r_edge))]],
                    [r_edge, r_edge], "k-", linewidth=0.8)

        # 画中间表面（胶合面）
        for mi in middle_surfs:
            z_mid = _surface_sag_profile(surfaces[mi], rr)
            ax.plot(z_mid, rr, "k-", linewidth=0.8)

        # 标注材料
        z_mid_pos = (z_front[len(z_front) // 2] + z_back[len(z_back) // 2]) / 2
        mat_label = "/".join(set(mats[:-1]))  # 去重（胶合镜可能有多种材料）
        ax.text(z_mid_pos, r_ap + 1.5, mat_label, fontsize=7, ha="center", va="bottom",
                color="navy", fontweight="bold")

    # ---- 2. 绘制光阑 ----
    for surf in surfaces:
        if surf.type == "光阑":
            stop_z = surf.z_vertex
            stop_r = R * 0.8  # 光阑显示高度
            # 光阑标志：带箭头的短线
            ax.plot([stop_z, stop_z], [-stop_r, stop_r], "k-", linewidth=2.5)
            ax.plot([stop_z - 1.5, stop_z + 1.5], [stop_r, stop_r], "k-", linewidth=2.0)
            ax.plot([stop_z - 1.5, stop_z + 1.5], [-stop_r, -stop_r], "k-", linewidth=2.0)
            ax.text(stop_z, stop_r + 1.5, "STOP", fontsize=8, ha="center",
                    fontweight="bold", color="red")

    # ---- 3. 绘制像面 ----
    for surf in surfaces:
        if surf.type == "像面":
            img_z = surf.z_vertex
            img_r = R * 0.6
            ax.plot([img_z, img_z], [-img_r, img_r], "k-", linewidth=2.0)
            # 像面标记斜线
            for r_val in np.linspace(-img_r, img_r, 15):
                ax.plot([img_z - 0.5, img_z + 0.5], [r_val - 0.3, r_val + 0.3],
                        "k-", linewidth=0.4)
            ax.text(img_z, img_r + 1.5, "IMA", fontsize=8, ha="center",
                    fontweight="bold", color="green")

    # ---- 4. 光线追迹 ----
    field_colors = {0: "#1f77b4", 5: "#ff7f0e", 8: "#2ca02c", 10: "#d62728",
                    15: "#9467bd", 20: "#8c564b"}
    for view in views:
        color = field_colors.get(view, "gray")
        theta = np.radians(view)
        direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
        direction = direction / np.linalg.norm(direction)

        # 在入瞳上采样光线
        for x0 in np.linspace(-R * 0.9, R * 0.9, M):
            result = trace_ray((x0, 0.0, 0.0), direction, surfaces, n_func)
            pts = result["points"]
            if len(pts) >= 2:
                zs = [p[2] for p in pts]
                xs = [p[0] for p in pts]
                label = f"{view}°" if abs(x0 - (-R * 0.9)) < 1e-6 else None
                ax.plot(zs, xs, color=color, linewidth=0.5, alpha=0.75,
                        label=label)

    # ---- 5. 标注与美化 ----
    ax.set_xlabel("Z [mm]", fontsize=12)
    ax.set_ylabel("X [mm]", fontsize=12)
    ax.set_title("Optical System Layout — Lens Cross-Section with Ray Traces",
                 fontsize=13, fontweight="bold")

    # 设置 z 轴范围
    all_z = [s.z_vertex for s in surfaces]
    z_min, z_max = min(all_z) - 5, max(all_z) + 5
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-R * 1.5, R * 1.5)
    ax.set_aspect("equal")

    # 图例
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9, title="Field Angle")

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"已保存: {savepath}")

    return fig, ax


def plot_spot_diagram(image_pts, title="Spot Diagram"):
    """绘制点列图"""
    if len(image_pts) == 0:
        print("无有效像面交点，无法绘制点列图")
        return
    centroid = image_pts.mean(axis=0)
    dx = image_pts[:, 0] - centroid[0]
    dy = image_pts[:, 1] - centroid[1]
    rms = np.sqrt(np.mean(dx ** 2 + dy ** 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dx * 1000, dy * 1000, s=4, c="blue", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title(f"{title}\nRMS spot radius = {rms * 1000:.2f} μm")
    ax.set_aspect("equal")
    lim = max(abs(dx).max(), abs(dy).max()) * 1000 * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    return fig, ax, rms


# ==================== 主程序 ====================
if __name__ == "__main__":
    lens_file = "副本附件1_镜头参数.csv"
    material_file = "材料参数.csv"
    wavelength = 0.5876  # d-line [μm]

    print("=" * 50)
    print("加载光学系统...")
    surfaces, n_func = load_optical_system(lens_file, material_file, wavelength)
    for i, s in enumerate(surfaces):
        print(f"  表面{i}: {s.type:8s}  z={s.z_vertex:10.4f}  "
              f"材料={s.material:10s}  r={s.radius}")

    # 找到光阑半径
    print("\n计算光阑半径...")
    ref = trace_ray((0.0, 12.5, 0.0), (0.0, 0.0, 1.0), surfaces, n_func)
    R_stop = ref["stop_radius"]
    print(f"光阑半径 = {R_stop:.4f} mm")

    # ---- 1. 光学系统布局图（填充镜片 + 光线追迹） ----
    print("\n绘制光学系统布局图...")
    plot_lens_layout(surfaces, n_func, views=[0, 5, 10], R=12.5, M=7,
                     savepath="lens_layout.png")
    print("已保存 lens_layout.png")

    # ---- 2. 点列图（轴上，0°视场） ----
    print("\n追迹轴上光束 (0°)...")
    pts_0, _, _ = trace_grid(surfaces, n_func, view_angle_deg=0.0, N=15)
    plot_spot_diagram(pts_0, title="Spot Diagram — 0° field")
    plt.savefig("spot_0deg.png", dpi=150)
    print(f"  有效光线数: {len(pts_0)}")

    # ---- 3. 点列图（轴外，8°视场） ----
    print("\n追迹轴外光束 (8°)...")
    pts_8, _, _ = trace_grid(surfaces, n_func, view_angle_deg=8.0, N=15)
    plot_spot_diagram(pts_8, title="Spot Diagram — 8° field")
    plt.savefig("spot_8deg.png", dpi=150)
    print(f"  有效光线数: {len(pts_8)}")

    plt.show()
    print("\n完成!")
