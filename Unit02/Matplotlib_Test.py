from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("matplotlib_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_and_close(filename: str) -> None:
    """儲存目前圖表並關閉，避免圖表互相影響。"""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    # plt.show()  # 若想在本機跳出視窗，可取消註解
    plt.close()
    print(f"已輸出：{path}")


# ============================================================
# 1. 長條圖 bar chart
# ============================================================

def practice_bar_chart():
    labels = ["Python", "C++", "Java", "JS", "C", "C#"]
    index = np.arange(len(labels))
    ratings = [5.16, 5.73, 14.99, 3.17, 11.86, 4.45]

    plt.figure(figsize=(8, 4))
    plt.bar(index, ratings)
    plt.xticks(index, labels)
    plt.ylabel("Rating")
    plt.title("Programming Rating")
    save_and_close("01_bar_chart.png")


# ============================================================
# 2. 水平長條圖 horizontal bar chart
# ============================================================

def practice_horizontal_bar_chart():
    labels = ["Python", "C++", "Java", "JS", "C", "C#"]
    index = np.arange(len(labels))
    change = [1.12, 0.3, -1.69, 0.29, 3.41, -0.45]

    plt.figure(figsize=(8, 4))
    plt.barh(index, change)
    plt.yticks(index, labels)
    plt.xlabel("Change")
    plt.title("Programming Rating Change")
    save_and_close("02_horizontal_bar_chart.png")


# ============================================================
# 3. 直方圖 histogram
# ============================================================

def practice_histogram():
    x = [21, 42, 23, 4, 5, 26, 77, 88, 9, 10,
         31, 32, 33, 34, 35, 36, 37, 18, 49, 50, 100]

    num_bins = 5

    plt.figure(figsize=(8, 4))
    n, bins, patches = plt.hist(x, num_bins)

    print("histogram 每個區間的數量 n：")
    print(n)
    print("histogram 區間 bins：")
    print(bins)
    print("patches 物件：")
    print(patches)

    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")
    save_and_close("03_histogram.png")


# ============================================================
# 4. 雙 y 軸圖 twinx
# ============================================================

def practice_twin_axis():
    x = np.linspace(0, 10, 50)
    sinus = np.sin(x)
    sinhs = np.sinh(x)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, sinus, "r-o", label="sin(x)")
    ax.set_ylabel("sin(x)")

    ax2 = ax.twinx()
    ax2.plot(x, sinhs, "g--", label="sinh(x)")
    ax2.set_ylabel("sinh(x)")

    plt.title("sin(x) and sinh(x)")
    save_and_close("04_twin_axis.png")


# ============================================================
# 5. 標準函數畫圖：y = sin(x)
# ============================================================

def practice_plot_sin():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("y = sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    save_and_close("05_sin_function.png")


# ============================================================
# 6. 練習函數畫圖：f(x) = sin(5x) / (1 + x^2)
# ============================================================

def practice_custom_function():
    x = np.linspace(-10, 10, 300)
    y = np.sin(5 * x) / (1 + x ** 2)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("f(x) = sin(5x) / (1 + x^2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    save_and_close("06_custom_function.png")


# ============================================================
# 7. 快速改變顏色、線條風格
# ============================================================

def practice_line_style():
    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), "r--", label="red dashed")
    plt.plot(x, np.cos(x), "g-.", label="green dash dot")
    plt.plot(x, np.sin(x) + np.cos(x), "bo", markevery=8, label="blue circles")
    plt.title("Line Colors and Styles")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    save_and_close("07_line_style.png")


# ============================================================
# 8. 基本修飾：alpha、color、linestyle、linewidth
# ============================================================

def practice_line_setting():
    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(8, 4))
    plt.plot(
        x,
        np.sin(x),
        alpha=0.7,
        color="purple",
        linestyle="--",
        linewidth=3,
        label="sin(x)",
    )
    plt.plot(
        x,
        np.cos(x),
        alpha=0.7,
        color="orange",
        linestyle="-.",
        linewidth=3,
        label="cos(x)",
    )
    plt.title("Basic Line Settings")
    plt.legend()
    plt.grid(True)
    save_and_close("08_line_setting.png")


# ============================================================
# 9. 參數式圖形：畫圓
# ============================================================

def practice_parametric_circle():
    t = np.linspace(-2 * np.pi, 2 * np.pi, 300)
    r = 1

    x = r * np.cos(t)
    y = r * np.sin(t)

    plt.figure(figsize=(5, 5))
    plt.plot(x, y)
    plt.axis("equal")  # 讓 x、y 比例一致，圓才不會變成橢圓
    plt.title("Parametric Circle")
    plt.grid(True)
    save_and_close("09_parametric_circle.png")


# ============================================================
# 10. 參數式圖形：半徑會變化的圖形
# ============================================================

def practice_parametric_curve():
    t = np.linspace(0, 20 * np.pi, 2000)

    r = 1 + 0.5 * np.sin(6 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y)
    plt.axis("equal")
    plt.title("Parametric Curve")
    plt.grid(True)
    save_and_close("10_parametric_curve.png")


# ============================================================
# 11. subplot：一張 figure 裡面放多張圖
# ============================================================

def practice_subplot():
    x = np.linspace(-10, 10, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title("sin(x)")

    axes[0, 1].plot(x, np.cos(x))
    axes[0, 1].set_title("cos(x)")

    axes[1, 0].plot(x, x ** 2)
    axes[1, 0].set_title("x^2")

    axes[1, 1].plot(x, np.exp(-x ** 2))
    axes[1, 1].set_title("exp(-x^2)")

    save_and_close("11_subplot.png")


# ============================================================
# 12. 進階色彩
# ============================================================

def practice_color_expression():
    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), c="r", label="short color: r")
    plt.plot(x, np.sin(x) + 1, c="0.6", label="gray scale: 0.6")
    plt.plot(x, np.sin(x) + 2, c="#00a676", label="hex color")
    plt.plot(x, np.sin(x) + 3, c=(0.7, 0.4, 1), label="RGB tuple")
    plt.title("Different Color Expressions")
    plt.legend()
    save_and_close("12_color_expression.png")


# ============================================================
# 13. Marker 標記
# ============================================================

def practice_marker():
    x = np.linspace(0, 10, 15)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(
        x,
        y,
        marker="o",
        markeredgecolor="black",
        markeredgewidth=2,
        markerfacecolor="yellow",
        markersize=10,
        linewidth=2,
    )
    plt.title("Marker Settings")
    plt.grid(True)
    save_and_close("13_marker.png")


# ============================================================
# 14. markevery：每隔幾個點才畫 marker
# ============================================================

def practice_markevery():
    x = np.linspace(0, 20, 300)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", markevery=10)
    plt.title("markevery=10")
    plt.grid(True)
    save_and_close("14_markevery.png")


# ============================================================
# 15. 雙長條圖
# ============================================================

def practice_double_bar_chart():
    labels = ["A", "B", "C", "D", "E"]
    x = np.arange(len(labels))

    group_1 = np.array([10, 20, 15, 30, 25])
    group_2 = np.array([12, 18, 20, 28, 22])

    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, group_1, width, label="Group 1")
    plt.bar(x + width / 2, group_2, width, label="Group 2")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Double Bar Chart")
    plt.legend()
    save_and_close("15_double_bar_chart.png")


# ============================================================
# 16. 疊加長條圖 stacked bar chart
# ============================================================

def practice_stacked_bar_chart():
    labels = ["A", "B", "C", "D", "E"]
    x = np.arange(len(labels))

    math = np.array([80, 70, 90, 60, 85])
    english = np.array([75, 88, 65, 70, 90])

    plt.figure(figsize=(8, 4))
    plt.bar(x, math, label="Math")
    plt.bar(x, english, bottom=math, label="English")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Stacked Bar Chart")
    plt.legend()
    save_and_close("16_stacked_bar_chart.png")


# ============================================================
# 17. 雙向水平長條圖
# ============================================================

def practice_bidirectional_bar_chart():
    labels = ["Python", "C++", "Java", "JS", "C", "C#"]
    y = np.arange(len(labels))

    positive = np.array([5.16, 5.73, 14.99, 3.17, 11.86, 4.45])
    negative = -np.array([1.12, 0.3, 1.69, 0.29, 3.41, 0.45])

    plt.figure(figsize=(8, 4))
    plt.barh(y, positive, label="Rating")
    plt.barh(y, negative, label="Change")

    plt.yticks(y, labels)
    plt.axvline(0, linewidth=1)
    plt.title("Bidirectional Horizontal Bar Chart")
    plt.legend()
    save_and_close("17_bidirectional_bar_chart.png")


# ============================================================
# 18. 圖表文字標示 annotate
# ============================================================

def practice_annotation():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    max_index = np.argmax(y)
    max_x = x[max_index]
    max_y = y[max_index]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.scatter([max_x], [max_y])

    plt.annotate(
        "maximum",
        xy=(max_x, max_y),
        xytext=(max_x + 1, max_y - 0.5),
        arrowprops={"arrowstyle": "->"},
    )

    plt.title("Annotation Example")
    plt.grid(True)
    save_and_close("18_annotation.png")


# ============================================================
# 19. 邊界設定 xlim / ylim
# ============================================================

def practice_axis_limit():
    x = np.linspace(-10, 10, 300)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlim(-5, 5)
    plt.ylim(-1.2, 1.2)
    plt.title("xlim and ylim")
    plt.grid(True)
    save_and_close("19_axis_limit.png")


# ============================================================
# 20. xticks：自訂 X 軸標籤
# ============================================================

def practice_xticks():
    x = np.arange(1, 6)
    y = np.array([90, 85, 75, 88, 92])
    labels = ["甲", "乙", "丙", "丁", "戊"]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o")
    plt.xticks(x, labels)
    plt.xlabel("學生")
    plt.ylabel("分數")
    plt.title("自訂 xticks")
    plt.grid(True)
    save_and_close("20_xticks.png")


# ============================================================
# 21. legend 圖例
# ============================================================

def practice_legend():
    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), label="sin(x)")
    plt.plot(x, np.cos(x), label="cos(x)")
    plt.title("Legend Example")
    plt.legend(loc=1)
    plt.grid(True)
    save_and_close("21_legend.png")


# ============================================================
# 22. 取得 axes 並設定背景顏色
# ============================================================

def practice_gca_axes():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)

    ax = plt.gca()
    ax.set_facecolor("#eeeeee")

    plt.title("Get Current Axes")
    plt.grid(True)
    save_and_close("22_gca_axes.png")


# ============================================================
# 23. 移動 x、y 座標軸
# ============================================================

def practice_move_axis():
    x = np.linspace(-10, 10, 400)
    y = x ** 3 - 5 * x

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)

    ax = plt.gca()

    # 把左邊與下面座標軸移到資料座標 0 的位置
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))

    # 隱藏右邊與上面的邊框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    plt.title("Move x-axis and y-axis")
    plt.grid(True)
    save_and_close("23_move_axis.png")


# ============================================================
# 24. xkcd 風格
# ============================================================

def practice_xkcd():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    with plt.xkcd():
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title("xkcd Style")
        plt.xlabel("x")
        plt.ylabel("sin(x)")
        save_and_close("24_xkcd.png")


# ============================================================
# 25. Seaborn：搭配 Matplotlib 美化
# ============================================================

def practice_seaborn_optional():
    try:
        import seaborn as sns
    except ImportError:
        print("尚未安裝 seaborn，略過 seaborn 範例。")
        print("可使用：pip install seaborn")
        return

    sns.set_theme()

    x = np.linspace(0, 10, 100)

    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), label="sin(x)")
    plt.plot(x, np.cos(x), label="cos(x)")
    plt.title("Seaborn Theme with Matplotlib")
    plt.legend()
    save_and_close("25_seaborn_theme.png")


# ============================================================
# 主程式
# ============================================================

def main():
    practice_bar_chart()
    practice_horizontal_bar_chart()
    practice_histogram()
    practice_twin_axis()

    practice_plot_sin()
    practice_custom_function()
    practice_line_style()
    practice_line_setting()

    practice_parametric_circle()
    practice_parametric_curve()
    practice_subplot()
    practice_color_expression()

    practice_marker()
    practice_markevery()

    practice_double_bar_chart()
    practice_stacked_bar_chart()
    practice_bidirectional_bar_chart()

    practice_annotation()
    practice_axis_limit()
    practice_xticks()
    practice_legend()

    practice_gca_axes()
    practice_move_axis()
    practice_xkcd()
    practice_seaborn_optional()

    print(f"圖檔輸出資料夾：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
