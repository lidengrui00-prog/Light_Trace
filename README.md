# 1.介绍
这是一个简单的光学追迹程序，镜头的参数和相关材料保存在
> Task_trace\Lens_parameter.csv
> Task_trace\material_parameter.csv

对于材料中的sellmeier_1类型，计算折射率应该使用公式：

$$
n^2 - 1 = \frac{K_1\lambda^2}{\lambda^2 - L_1} + \frac{K_2\lambda^2}{\lambda^2 - L_2} + \frac{K_3\lambda^2}{\lambda^2 - L_3}
$$
而对于材料中的schott类型，应该使用公式：
$$
n^2 = a_0 + a_1\lambda^2 + a_2\lambda^{-2} + a_3\lambda^{-4} + a_4\lambda^{-6} + a_5\lambda^{-8}
$$
# 2.代码解析
在追迹过程中，由于只用到了两种曲面，我只在代码中定义了球面和XY多项式曲面。系统的入瞳直接是25mm，代码中先追迹了起点为(0.0, 12.5, 0.0)，方向为(0, 0, 1)光线，来得到光阑半径，之后在根据光阑半径，用坐标下降法来得到各个视场下的光线起点。
## 2.1 曲面与光线交点
对于简单的球面镜，如图1所示，求交点的方法使用二次求根法。
<p align="center">
  <img src="Task_trace/Picture_instructions/Sphere.drawio.png" alt="球面示意图"><br>
  <sub>图1：球面示意图</sub>
</p>
设曲线的公式为
