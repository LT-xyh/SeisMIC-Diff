import os

import numpy as np
from matplotlib import pyplot as plt


def save_visualize_image(img, filename='', title="Image", show=False, save=True, cmap='jet', extent=[0, 700, 700, 0],
                         figsize=(5, 5), use_colorbar=True, x_label='Length (m)', y_label='Depth (m)'):
    """
    保存单个灰度图像到文件
    :param img: 要显示的图像
    :param filename: 保存的文件名(带路径)
    :param title: 图像标题
    :param show: 是否显示图像
    :param save: 是否保存图像
    :param cmap: 颜色映射，默认为'jet'
    :param extent: 图像显示范围
    :param figsize: 图像尺寸，元组(宽度, 高度)
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(img, aspect='auto', cmap=cmap, extent=extent, vmax=1, vmin=-1)
    # ax.set_title(title)
    ax.set_xlabel(x_label, loc='center')
    ax.set_ylabel(y_label, loc='center')
    # ax.set_xticks(extent[:2])
    # ax.set_yticks(extent[2:])
    if use_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        original_ticks = np.linspace(0, 1, 5)  # 生成5个均匀分布的刻度
        target_ticks = np.linspace(1500, 4500, 5)
        cbar.set_ticks(original_ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in target_ticks])
        cbar.set_label('Velocity (m/s)')  # 设置颜色条的标签

    # plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save:  # 保存图像
        plt.savefig(filename)

    if show:  # 显示图像(阻塞模式)
        plt.show()

    # 关闭图形以释放内存
    plt.close(fig)


def save_multiple_curves(curves, labels=None, filename='', title="Well log", x_label="Depth (m)",
                         y_label="Velocity (m/s)", show=True, save=False, figsize=(6, 6), colors=None, linestyles=None):
    """
    在同一图表中可视化多条曲线

    参数:
    curves : list of arrays
        要绘制的曲线数据列表，每个元素是一条曲线的y值
    labels : list of str, optional
        每条曲线的标签，用于图例显示
    filename : str
        保存图像的文件名(带路径)
    title : str
        图表标题
    x_label : str
        x轴标签
    y_label : str
        y轴标签
    show : bool
        是否显示图像
    save : bool
        是否保存图像
    figsize : tuple
        图像尺寸 (宽度, 高度)
    colors : list of str, optional
        每条曲线的颜色
    linestyles : list of str, optional
        每条曲线的线型

    返回:
    None
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 如果没有提供标签，则生成默认标签
    if labels is None:
        # labels = [f'Well log {i+1}' for i in range(len(curves))]
        labels = ['Well log']

    # 如果没有提供颜色，则使用默认颜色循环
    if colors is None:
        colors = [None] * len(curves)

    # 如果没有提供线型，则使用默认线型
    if linestyles is None:
        linestyles = ['-'] * len(curves)

    # 绘制每条曲线
    for i, (curve, label, color, linestyle) in enumerate(zip(curves, labels, colors, linestyles)):
        x_values = range(0, len(curve) * 10, 10)  # 自动生成x值
        ax.plot(x_values, curve, label=label, color=color, linestyle=linestyle, linewidth=2)

    # 设置图表属性
    # ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    # ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # 保存图像
    if save:
        plt.savefig(filename)

    # 显示图像
    if show:
        plt.show()

    # 关闭图形以释放内存
    plt.close(fig)


def test_visualize(dataset_name):
    save_path = f"images/datasets/"
    os.makedirs(save_path, exist_ok=True)
    show = True
    save = False
    depth_vel = np.load(f'data/openfwi/{dataset_name}/depth_vel/27006.npy').squeeze()
    depth_vel = (depth_vel - depth_vel.min()) / (depth_vel.max() - depth_vel.min())
    save_visualize_image(depth_vel, filename=os.path.join(save_path, f'{dataset_name}_depth_vel.svg'),
                         title="Depth Velocity", show=show, save=save, cmap='jet', extent=[0, 700, 700, 0],
                         figsize=(5, 5), use_colorbar=True, x_label='Length (m)', y_label='Depth (m)',)
    rms_vel = np.load(f'data/openfwi/{dataset_name}/rms_vel/27006.npy').squeeze()
    save_visualize_image(rms_vel, filename=os.path.join(save_path, f'{dataset_name}_time _vel.svg'), title="Time Velocity",
                         show=show, save=save, cmap='jet', extent=[0, 700, 1, 0], figsize=(5, 5), use_colorbar=False,
                         x_label='Length (m)', y_label='Time (s)')
    migrated_image = np.load(f'data/openfwi/{dataset_name}/migrated_image/27006.npy').squeeze()
    save_visualize_image(migrated_image, filename=os.path.join(save_path, f'{dataset_name}_migrated_image.svg'),
                         title="Migrated Image", show=show, save=save, cmap='gray', extent=[0, 700, 1, 0],
                         figsize=(5, 5), use_colorbar=False, x_label='Length (m)', y_label='Time (s)')

    horizon = np.load(f'data/openfwi/{dataset_name}/horizon/27006.npy').squeeze()
    save_visualize_image(horizon, filename=os.path.join(save_path, f'{dataset_name}_horizon.svg'), title="Horizon",
                         show=show, save=save, cmap='gray', extent=[0, 700, 700, 0], figsize=(5, 5), use_colorbar=False,
                         x_label='Length (m)', y_label='Depth (m)')
    well_log = depth_vel[:, 20]
    save_multiple_curves([well_log], filename=os.path.join(save_path, f'{dataset_name}_well_log.svg'), show=show,
                         save=save, )


if __name__ == '__main__':
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        test_visualize(dataset_name)
        break
