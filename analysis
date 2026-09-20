"""
Анализ A/B-теста новой кнопки.

Читает data/ab_test.csv, считает всю статистику, печатает ключевые цифры,
сохраняет график в outputs/ и числовой отчёт в outputs/results.txt.

Используются только pandas / numpy / matplotlib + стандартная библиотека.
Стат-функции (z-тест, хи-квадрат, нормальное распределение) реализованы вручную,
чтобы не зависеть от scipy.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # без дисплея, рендерим в файл
import matplotlib.pyplot as plt

HERE = "/content"
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
ALPHA = 0.05


# ---------- вспомогательные стат-функции (без scipy) ----------

def norm_cdf(x):
    """Функция распределения стандартного нормального через erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    """Двусторонний p-value для z-статистики."""
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def chi2_sf_df1(x):
    """
    P(Chi2_1 > x) для 1 степени свободы.
    Для df=1 хи-квадрат = z^2, поэтому хвост = 2*(1 - Phi(sqrt(x))).
    """
    if x <= 0:
        return 1.0
    return 2.0 * (1.0 - norm_cdf(math.sqrt(x)))


# ---------- основной анализ ----------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "ab_test.csv"))

    a = df[df["group"] == "A"]
    b = df[df["group"] == "B"]

    n_a, n_b = len(a), len(b)
    conv_a, conv_b = int(a["converted"].sum()), int(b["converted"].sum())
    p_a, p_b = conv_a / n_a, conv_b / n_b

    # --- Sample Ratio Mismatch: ожидаем 50/50, проверяем хи-квадратом ---
    n_total = n_a + n_b
    exp = n_total / 2.0
    srm_chi2 = (n_a - exp) ** 2 / exp + (n_b - exp) ** 2 / exp
    srm_p = chi2_sf_df1(srm_chi2)
    srm_ok = srm_p >= 0.05  # ok, если НЕ значимо (распределение здоровое)

    # --- Z-тест разницы двух пропорций (pooled) ---
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    z = diff / se_pool
    p_value = two_sided_p_from_z(z)

    # --- Хи-квадрат на таблице 2x2 (с поправкой Йейтса) — для сверки ---
    conv = np.array([[conv_a, n_a - conv_a],
                     [conv_b, n_b - conv_b]], dtype=float)
    row = conv.sum(axis=1)
    col = conv.sum(axis=0)
    expected = np.outer(row, col) / n_total
    chi2_stat = ((np.abs(conv - expected) - 0.5) ** 2 / expected).sum()
    chi2_p = chi2_sf_df1(chi2_stat)

    # --- 95% доверительный интервал разницы (unpooled SE) ---
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = diff - 1.96 * se_unpooled
    ci_high = diff + 1.96 * se_unpooled

    # --- Аплифты ---
    abs_uplift = diff
    rel_uplift = diff / p_a

    significant = p_value < ALPHA

    # --- ДИ для каждой группы (для графика, Wald) ---
    ci_a = 1.96 * math.sqrt(p_a * (1 - p_a) / n_a)
    ci_b = 1.96 * math.sqrt(p_b * (1 - p_b) / n_b)

    # ---------- печать ----------
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 56)
    out("A/B-ТЕСТ: новая кнопка повышает конверсию?")
    out("=" * 56)
    out()
    out(f"Группа A: {n_a:>6} польз., {conv_a:>5} конверсий, CR = {p_a*100:.2f}%")
    out(f"Группа B: {n_b:>6} польз., {conv_b:>5} конверсий, CR = {p_b*100:.2f}%")
    out()
    out("--- Sample Ratio Mismatch (ожидали 50/50) ---")
    out(f"chi2 = {srm_chi2:.3f}, p = {srm_p:.4f} -> "
        f"{'распределение здоровое (OK)' if srm_ok else 'ВНИМАНИЕ: перекос групп!'}")
    out()
    out("--- Разница конверсий ---")
    out(f"Абсолютный аплифт: {abs_uplift*100:+.2f} п.п.")
    out(f"Относительный аплифт: {rel_uplift*100:+.2f}%")
    out(f"95% ДИ разницы: [{ci_low*100:+.2f} п.п.; {ci_high*100:+.2f} п.п.]")
    out()
    out("--- Проверка значимости ---")
    out(f"Z-тест двух пропорций: z = {z:.3f}, p-value = {p_value:.5f}")
    out(f"Хи-квадрат 2x2 (Йейтс): chi2 = {chi2_stat:.3f}, p-value = {chi2_p:.5f}")
    out(f"Уровень значимости alpha = {ALPHA}")
    out()
    if significant:
        out(f"ВЫВОД: разница СТАТИСТИЧЕСКИ ЗНАЧИМА (p = {p_value:.5f} < {ALPHA}).")
        out("Новая кнопка реально повышает конверсию — можно раскатывать на всех.")
    else:
        out(f"ВЫВОД: разница НЕ значима (p = {p_value:.5f} >= {ALPHA}).")
        out("Данных недостаточно, чтобы утверждать эффект. Не раскатываем.")
    out("=" * 56)

    # ---------- график ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = ["A (старая)", "B (новая)"]
    means = [p_a * 100, p_b * 100]
    errs = [ci_a * 100, ci_b * 100]
    colors = ["#9aa7b1", "#e0a458"]

    bars = ax.bar(groups, means, yerr=errs, capsize=10,
                  color=colors, edgecolor="#444", linewidth=1.2,
                  error_kw={"elinewidth": 1.5, "ecolor": "#444"})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(errs) * 0.15,
                f"{m:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Конверсия, %")
    ax.set_title("Конверсия по группам (95% ДИ)\n"
                 f"абс. аплифт {abs_uplift*100:+.2f} п.п., p = {p_value:.4f}")
    ax.set_ylim(0, max(means) + max(errs) * 2.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_path = os.path.join(OUT_DIR, "conversion_by_group.png")
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
    out()
    out(f"График сохранён: {chart_path}")

    # ---------- results.txt ----------
    res_path = os.path.join(OUT_DIR, "results.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Отчёт сохранён: {res_path}")

    # вернём числа для README/meta
    return {
        "n_a": n_a, "n_b": n_b,
        "p_a": p_a, "p_b": p_b,
        "srm_chi2": srm_chi2, "srm_p": srm_p,
        "z": z, "p_value": p_value,
        "ci_low": ci_low, "ci_high": ci_high,
        "abs_uplift": abs_uplift, "rel_uplift": rel_uplift,
        "significant": significant,
    }


if __name__ == "__main__":
    main()
