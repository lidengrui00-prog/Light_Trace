from opticsTrace import *
import numpy as np
import pandas as pd

lens_file = r"C:\python_work\光线追迹\Task_trace\Lens_parameter.csv"
material_file = r"C:\python_work\光线追迹\Task_trace\material_parameter.csv"

wavelengths = [0.656273, 0.4861327, 0.5875618]
colors = ['red',   'blue', 'green']
markers = ['^',     '+',    's']
labels = ['0.656273 μm', '0.4861327 μm', '0.5875618 μm']

theta = math.radians(8.0)
phi = math.radians(3.0)
dir_z = 1/math.sqrt(math.tan(theta)**2 + math.tan(phi)**2 +1)
dir_x = math.tan(theta) * dir_z
dir_y = math.tan(phi) * dir_z
direction = np.array([dir_x, dir_y, dir_z])

all_image_points = {}
all_results = {}

for wavelength in wavelengths:

    print(f"\n{'='*60}")
    print(f"Trace the wavelength: {wavelength} μm")
    print(f"{'='*60}")

    surfaces, n_func = load_optical_system(lens_file,material_file,wavelength)

    #得到光阑半径
    ref_result = trace_ray(np.array([0.0,12.5,0.0]), np.array([0.0,0.0,1.0]),surfaces, n_func)
    R = ref_result["stop_radius"]
    print(f"The stop radius is {R:.4f}mm")

    #六级环采样
    a, b = hexapolar_rings(density=6)
    pupil_xy = torch.cat([a,b], dim=1)
    pupil_xy = pupil_xy.numpy() * R
    print(f"All get {len(pupil_xy)} sample points")

    #逐个优化并追迹
    image_points_xy = []
    success_count = 0
    total = len(pupil_xy)

    for i, target in enumerate(pupil_xy):
        target_tuple = (target[0], target[1])
        start_pt, success, err, iters = optimize_coordinate_descent(
            target_tuple, direction, surfaces, n_func)
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

        ps_tensor = torch.tensor(image_points_xy)
        rms, geo, centroid = spot_analysis(ps_tensor, option='centroid')

        cx, cy = centroid[0].item(), centroid[1].item()
        image_points_xy[:, 0] -= cx
        image_points_xy[:, 1] -= cy
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

plt.figure(figsize=(5, 5))
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

for wl, color, marker, label in zip(wavelengths, colors, markers, labels):
    pts = all_image_points.get(wl)
    if pts is not None:
        plt.scatter(pts[:, 0], pts[:, 1],color=color, marker=marker, s=12, zorder=5,label=label)

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Spot Diagram (Hexapolar Rings, 3 wavelengths)")
plt.axis("equal")
# 自动设置坐标轴范围：取最大 GEO 半径的 1.5 倍，最小 0.02 mm
max_geo = max([r[1] for r in all_results.values()])
lim = max(max_geo * 1.5, 0.02)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()