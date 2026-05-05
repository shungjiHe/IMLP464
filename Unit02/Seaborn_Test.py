import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("seaborn_output")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans")


# ------------------------------------------------------------
# 共用工具
# ------------------------------------------------------------
def save_plot(filename: str) -> None:
    """儲存目前的圖表，然後關閉 figure。"""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def show_head(name: str, df: pd.DataFrame, n: int = 5) -> None:
    """印出 DataFrame 前幾筆，方便學生確認資料內容。"""
    print(f"\n===== {name} =====")
    print(df.head(n))


def make_fake_nba_data() -> pd.DataFrame:
    """建立 NBA 薪資練習資料，避免外部 CSV 不存在時無法練習。"""
    players = [
        "Stephen Curry", "LeBron James", "Paul Millsap", "Gordon Hayward",
        "Blake Griffin", "Kyle Lowry", "Mike Conley", "James Harden",
        "DeMar DeRozan", "Russell Westbrook", "Kevin Durant", "Chris Paul",
        "Anthony Davis", "Al Horford", "Carmelo Anthony"
    ]

    teams = [
        "GSW", "CLE", "DEN", "BOS", "DET", "TOR", "MEM", "HOU",
        "TOR", "OKC", "GSW", "HOU", "NOP", "BOS", "OKC"
    ]

    salaries = [
        34.68, 33.29, 31.27, 29.73, 29.51, 28.70, 28.53, 28.30,
        27.74, 26.54, 25.00, 24.27, 23.78, 27.73, 26.24
    ]

    positions = [
        "PG", "SF", "PF", "SF", "PF", "PG", "PG", "SG",
        "SG", "PG", "SF", "PG", "PF", "C", "SF"
    ]

    return pd.DataFrame({
        "Player": players,
        "Team": teams,
        "Position": positions,
        "Salary": salaries,
    })


def make_fake_tips_data(seed: int = 42, n: int = 160) -> pd.DataFrame:
    """建立 tips 類型資料，避免 seaborn 內建資料集下載失敗。"""
    rng = np.random.default_rng(seed)

    days = rng.choice(["Thur", "Fri", "Sat", "Sun"], size=n, p=[0.25, 0.15, 0.35, 0.25])
    times = rng.choice(["Lunch", "Dinner"], size=n, p=[0.35, 0.65])
    sex = rng.choice(["Male", "Female"], size=n)
    smoker = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    size = rng.integers(1, 7, size=n)

    total_bill = rng.normal(18, 7, size=n) + size * 2.5
    total_bill = np.clip(total_bill, 5, 55).round(2)

    tip_rate = rng.normal(0.16, 0.04, size=n)
    tip = np.clip(total_bill * tip_rate, 1, 12).round(2)

    return pd.DataFrame({
        "total_bill": total_bill,
        "tip": tip,
        "sex": sex,
        "smoker": smoker,
        "day": days,
        "time": times,
        "size": size,
    })


def load_tips_data() -> pd.DataFrame:
    """建立 tips 類型資料；學生也可以自行改成 sns.load_dataset("tips")。"""
    return make_fake_tips_data(n=120)



def load_iris_data() -> pd.DataFrame:
    """建立 iris 類型資料；學生也可以自行改成 sns.load_dataset("iris")。"""
    rng = np.random.default_rng(42)
    species = np.repeat(["setosa", "versicolor", "virginica"], 35)

    sepal_length = np.concatenate([
        rng.normal(5.0, 0.3, 35),
        rng.normal(5.9, 0.4, 35),
        rng.normal(6.5, 0.5, 35),
    ])
    sepal_width = np.concatenate([
        rng.normal(3.4, 0.3, 35),
        rng.normal(2.8, 0.3, 35),
        rng.normal(3.0, 0.3, 35),
    ])
    petal_length = np.concatenate([
        rng.normal(1.5, 0.2, 35),
        rng.normal(4.2, 0.4, 35),
        rng.normal(5.5, 0.5, 35),
    ])
    petal_width = np.concatenate([
        rng.normal(0.25, 0.08, 35),
        rng.normal(1.3, 0.2, 35),
        rng.normal(2.0, 0.3, 35),
    ])

    return pd.DataFrame({
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
        "species": species,
    })



# ------------------------------------------------------------
# 2.4.1 color_palette.ipynb
# ------------------------------------------------------------
def practice_color_palette() -> None:
    print("\n[練習] Seaborn color palette")

    # 1. 預設調色盤
    current_palette = sns.color_palette()
    sns.palplot(current_palette)
    save_plot("01_default_palette.png")

    # 2. seaborn 提供的 6 種常用調色盤
    theme_list = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]

    for theme in theme_list:
        sns.palplot(sns.color_palette(theme))
        save_plot(f"02_palette_{theme}.png")

    # 3. hls 調色盤
    sns.palplot(sns.color_palette("hls", 25))
    save_plot("03_hls_25_palette.png")

    # 4. hls 搭配 boxplot
    data = np.random.normal(size=(20, 12)) + np.arange(12) / 2

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data, palette=sns.color_palette("hls", 12))
    plt.title("Boxplot with HLS Palette")
    save_plot("04_boxplot_hls_palette.png")

    # 5. Blues、Reds、Greens_r
    for palette_name in ["Blues", "Reds", "Greens_r"]:
        sns.palplot(sns.color_palette(palette_name))
        save_plot(f"05_palette_{palette_name}.png")

    # 6. hls_palette 亮度與飽和度
    sns.palplot(sns.hls_palette(3))
    save_plot("06_hls_palette_default.png")

    sns.palplot(sns.hls_palette(3, l=0.8, s=0.8))
    save_plot("07_hls_palette_light_saturation.png")

    # 7. 使用 xkcd_rgb 顏色畫線
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], color=sns.xkcd_rgb["purple"], lw=2, label="purple")
    plt.plot([0, 1], [0, 2], color=sns.xkcd_rgb["green"], lw=3, label="green")
    plt.plot([0, 1], [0, 3], color=sns.xkcd_rgb["orange"], lw=4, label="orange")
    plt.legend()
    plt.title("xkcd_rgb Colors")
    save_plot("08_xkcd_rgb_lines.png")

    # 8. cubehelix 調色盤
    sns.palplot(sns.color_palette("cubehelix"))
    save_plot("09_cubehelix_palette.png")

    sns.palplot(sns.cubehelix_palette(8, start=0.8, rot=-0.5))
    save_plot("10_cubehelix_custom_palette.png")

    # 9. light_palette / dark_palette
    sns.palplot(sns.light_palette("green"))
    save_plot("11_light_palette_green.png")

    sns.palplot(sns.dark_palette("green"))
    save_plot("12_dark_palette_green.png")

    # 10. KDE 搭配 dark_palette
    x, y = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, -0.5], [-0.5, 1]],
        size=600
    ).T

    plt.figure(figsize=(6, 5))
    pal = sns.dark_palette("green", as_cmap=True)
    sns.kdeplot(x=x, y=y, cmap=pal, fill=True)
    plt.title("KDE Plot with Dark Palette")
    save_plot("13_kde_dark_palette.png")

    # 11. KDE 搭配 cubehelix
    x, y = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, -0.5], [-0.5, 1]],
        size=600
    ).T

    plt.figure(figsize=(6, 5))
    pal = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.kdeplot(x=x, y=y, cmap=pal, fill=True)
    plt.title("KDE Plot with Cubehelix Palette")
    save_plot("14_kde_cubehelix_palette.png")


# ------------------------------------------------------------
# 2.4.2 Seaborn數據視覺化.ipynb
# ------------------------------------------------------------
def practice_nba_barplot() -> None:
    print("\n[練習] NBA 薪資資料：長條圖、分組彙總")

    nba = make_fake_nba_data()
    show_head("NBA Salary Data", nba)

    # 1. 依薪資排序，取前 10 名
    top10 = nba.sort_values(by="Salary", ascending=False).head(10)

    plt.figure(figsize=(11, 5))
    sns.barplot(data=top10, x="Player", y="Salary", errorbar=None, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 NBA Salaries")
    plt.ylabel("Salary: Million USD")
    save_plot("15_nba_top10_salary_barplot.png")

    # 2. 依球隊彙總薪資
    team_salary = (
        nba.groupby("Team", as_index=False)["Salary"]
        .sum()
        .sort_values(by="Salary", ascending=False)
    )

    show_head("Team Salary Summary", team_salary)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=team_salary, x="Team", y="Salary", errorbar=None, color="steelblue")
    plt.title("Team Salary Summary")
    plt.ylabel("Salary: Million USD")
    save_plot("16_nba_team_salary_barplot.png")


def practice_regression_and_relation_plots(tips: pd.DataFrame) -> None:
    print("\n[練習] relplot / regplot / lmplot 關係圖與回歸圖")
    show_head("Tips Data", tips)

    # 1. relplot：total_bill 和 tip 的關係
    g = sns.relplot(
        data=tips,
        x="total_bill",
        y="tip",
        hue="sex",
        style="smoker",
        size="size",
        sizes=(30, 180),
        height=5,
        aspect=1.3,
    )
    g.fig.suptitle("Tip vs Total Bill", y=1.03)
    g.savefig(OUTPUT_DIR / "17_relplot_tip_total_bill.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 2. regplot：total_bill 對 tip 的線性回歸
    plt.figure(figsize=(7, 5))
    sns.regplot(data=tips, x="total_bill", y="tip")
    plt.title("Regplot: Total Bill vs Tip")
    save_plot("18_regplot_total_bill_tip.png")

    # 3. lmplot：用 smoker 分組看回歸線
    g = sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker", height=5, aspect=1.3)
    g.fig.suptitle("Lmplot by Smoker", y=1.03)
    g.savefig(OUTPUT_DIR / "19_lmplot_smoker.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 4. 練習：size 和 tip 的關係
    plt.figure(figsize=(7, 5))
    sns.regplot(data=tips, x="size", y="tip", x_jitter=0.1)
    plt.title("Practice: Size vs Tip")
    save_plot("20_regplot_size_tip.png")


def practice_single_variable_plots(tips: pd.DataFrame) -> None:
    print("\n[練習] 單變量分析：直方圖、計數圖")

    # 1. total_bill 直方圖
    plt.figure(figsize=(7, 5))
    sns.histplot(data=tips, x="total_bill", bins=20, kde=True)
    plt.title("Histogram: Total Bill")
    save_plot("21_hist_total_bill.png")

    # 2. tip 直方圖
    plt.figure(figsize=(7, 5))
    sns.histplot(data=tips, x="tip", bins=20, kde=True)
    plt.title("Histogram: Tip")
    save_plot("22_hist_tip.png")

    # 3. 每一天的資料筆數
    plt.figure(figsize=(7, 5))
    sns.countplot(data=tips, x="day", color="steelblue")
    plt.title("Countplot: Day")
    save_plot("23_countplot_day.png")

    # 4. day 加上 sex 分組
    plt.figure(figsize=(7, 5))
    sns.countplot(data=tips, x="day", hue="sex")
    plt.title("Countplot: Day by Sex")
    save_plot("24_countplot_day_by_sex.png")


def practice_categorical_scatter(tips: pd.DataFrame) -> None:
    print("\n[練習] 分類散點圖：stripplot / swarmplot")

    # 1. stripplot：每一天的小費分布
    plt.figure(figsize=(7, 5))
    sns.stripplot(data=tips, x="day", y="tip", jitter=True)
    plt.title("Stripplot: Tip by Day")
    save_plot("25_stripplot_tip_day.png")

    # 2. stripplot 加上 hue
    plt.figure(figsize=(7, 5))
    sns.stripplot(data=tips, x="day", y="tip", hue="sex", jitter=True)
    plt.title("Stripplot: Tip by Day and Sex")
    save_plot("26_stripplot_tip_day_sex.png")

    # 3. swarmplot：避免點完全重疊
    plt.figure(figsize=(7, 5))
    sns.swarmplot(data=tips, x="day", y="tip", hue="sex")
    plt.title("Swarmplot: Tip by Day and Sex")
    save_plot("27_swarmplot_tip_day_sex.png")


def practice_box_violin_plots(tips: pd.DataFrame) -> None:
    print("\n[練習] 盒圖、小提琴圖、箱型分類圖")

    # 1. boxplot：用餐日與消費金額
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=tips, x="day", y="total_bill")
    plt.title("Boxplot: Total Bill by Day")
    save_plot("28_boxplot_total_bill_day.png")

    # 2. boxplot 加上 hue
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
    plt.title("Boxplot: Total Bill by Day and Sex")
    save_plot("29_boxplot_total_bill_day_sex.png")

    # 3. violinplot
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=tips, x="day", y="total_bill")
    plt.title("Violinplot: Total Bill by Day")
    save_plot("30_violinplot_total_bill_day.png")

    # 4. violinplot 加上 hue 與 split
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True)
    plt.title("Violinplot: Total Bill by Day and Sex")
    save_plot("31_violinplot_total_bill_day_sex.png")

    # 5. barplot：每一天平均小費
    plt.figure(figsize=(7, 5))
    sns.barplot(data=tips, x="day", y="tip", errorbar="sd", color="steelblue")
    plt.title("Barplot: Average Tip by Day")
    save_plot("32_barplot_avg_tip_day.png")

    # 6. pointplot：看平均值趨勢
    plt.figure(figsize=(7, 5))
    sns.pointplot(data=tips, x="day", y="tip", hue="sex", errorbar="sd")
    plt.title("Pointplot: Average Tip by Day and Sex")
    save_plot("33_pointplot_tip_day_sex.png")


def practice_multidimensional_plots(tips: pd.DataFrame, iris: pd.DataFrame) -> None:
    print("\n[練習] 多維資料展示：pairplot / heatmap / jointplot")

    # 1. pairplot：一次看多個欄位關係
    g = sns.pairplot(
        data=tips,
        vars=["total_bill", "tip", "size"],
        hue="sex",
        diag_kind="hist",
    )
    g.fig.suptitle("Pairplot: Tips", y=1.02)
    g.savefig(OUTPUT_DIR / "34_pairplot_tips.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 2. iris pairplot
    g = sns.pairplot(data=iris, hue="species", diag_kind="hist")
    g.fig.suptitle("Pairplot: Iris", y=1.02)
    g.savefig(OUTPUT_DIR / "35_pairplot_iris.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 3. heatmap：數值欄位相關係數
    numeric_tips = tips.select_dtypes(include=[np.number])

    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric_tips.corr(), annot=True, cmap="Blues")
    plt.title("Heatmap: Tips Correlation")
    save_plot("36_heatmap_tips_corr.png")

    # 4. jointplot：兩個變數的聯合分布
    g = sns.jointplot(data=tips, x="total_bill", y="tip", kind="scatter", height=6)
    g.fig.suptitle("Jointplot: Total Bill vs Tip", y=1.03)
    g.savefig(OUTPUT_DIR / "37_jointplot_total_bill_tip.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 5. jointplot KDE
    g = sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", fill=True, height=6)
    g.fig.suptitle("Jointplot KDE: Total Bill vs Tip", y=1.03)
    g.savefig(OUTPUT_DIR / "38_jointplot_kde_total_bill_tip.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def practice_facet_grid(tips: pd.DataFrame) -> None:
    print("\n[練習] FacetGrid：依分類切成多張小圖")

    # 1. 依 sex 與 time 切圖
    g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True, height=3)
    g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
    g.fig.suptitle("FacetGrid: Tips by Sex and Time", y=1.03)
    g.savefig(OUTPUT_DIR / "39_facetgrid_sex_time.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # 2. 依 day 切圖，畫 total_bill 分布
    g = sns.FacetGrid(tips, col="day", col_wrap=2, height=3.2)
    g.map_dataframe(sns.histplot, x="total_bill", bins=15, kde=True)
    g.fig.suptitle("FacetGrid: Total Bill by Day", y=1.03)
    g.savefig(OUTPUT_DIR / "40_facetgrid_total_bill_day.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def practice_style_and_context(tips: pd.DataFrame) -> None:
    print("\n[練習] seaborn 樣式與 context")

    # 1. darkgrid
    sns.set_style("darkgrid")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
    plt.title("Style: darkgrid")
    save_plot("41_style_darkgrid.png")

    # 2. white
    sns.set_style("white")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
    plt.title("Style: white")
    save_plot("42_style_white.png")

    # 3. talk context
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=tips, x="day", y="total_bill")
    plt.title("Context: talk")
    save_plot("43_context_talk.png")

    # 還原預設
    sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    np.random.seed(42)

    tips = load_tips_data()
    iris = load_iris_data()

    practice_color_palette()
    practice_nba_barplot()
    practice_regression_and_relation_plots(tips)
    practice_single_variable_plots(tips)
    practice_categorical_scatter(tips)
    practice_box_violin_plots(tips)
    practice_multidimensional_plots(tips, iris)
    practice_facet_grid(tips)
    practice_style_and_context(tips)

    print(f"\n完成，所有圖片已輸出到：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
