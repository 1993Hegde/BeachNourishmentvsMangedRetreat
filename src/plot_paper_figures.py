import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pickle
import colorcet as cc
from mordm_functions import *
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import math
import itertools
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

sns.set_style("whitegrid", {"axes_linewidth": 0, "axes.edgecolor": "white"})


def compare_2d_pareto_fronts_with_and_without_uncertainty(
    baseline_directory, uncertain_param_samples, all_parameters, initial_state
):
    base_path = os.path.join("..", "results_data")
    # File paths
    file1 = os.path.join(base_path, "npv_maximizing_strategy.pickle")
    file2 = os.path.join(base_path, "usace_strategy.pickle")
    with open(file1, "rb") as f:
        npv_maximizing_strategy = pickle.load(f)

    with open(file2, "rb") as f:
        usace_strategy = pickle.load(f)
    print("NPV mazimizing Strategy", npv_maximizing_strategy)
    print("USACE Strategy", usace_strategy)
    (max_npv_benefits, max_npv_costs), _ = (
        evaluate_beach_nourishment_problem_on_strategy_best_guess_sow(
            npv_maximizing_strategy, all_parameters
        )
    )
    (max_npv_benefits_across_sow, max_npv_costs_across_sow), _ = evaluate_individual(
        npv_maximizing_strategy, uncertain_param_samples, all_parameters, initial_state
    )
    print("Max NPV benefits:", max_npv_benefits, "Max NPV costs:", max_npv_costs)
    print(
        "Max NPV benefits across SoW:",
        max_npv_benefits_across_sow,
        "Max NPV costs across SoW:",
        max_npv_costs_across_sow,
    )

    (usace_benefits, usace_costs), _ = (
        evaluate_beach_nourishment_problem_on_strategy_best_guess_sow(
            usace_strategy, all_parameters
        )
    )
    (usace_benefits_across_sow, usace_costs_across_sow), _ = evaluate_individual(
        usace_strategy, uncertain_param_samples, all_parameters, initial_state
    )
    print("USACE benefits:", usace_benefits, "USACE costs:", usace_costs)
    print(
        "USACE benefits across SoW:",
        usace_benefits_across_sow,
        "USACE costs across SoW:",
        usace_costs_across_sow,
    )
    # Read in pareto front considering uncertainty
    pwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "2objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    os.chdir(final_path)
    with open("hof_fitness_across_sow.pkl", "rb") as handle:
        hof_fitness_across_sow = pickle.load(handle)
    with open("hof_across_sow.pkl", "rb") as handle:
        hof_across_sow = pickle.load(handle)
    df_hof_across_sow = pd.DataFrame(
        hof_fitness_across_sow, columns=["Benefits", "Costs"]
    )
    df_hof_across_sow["Strategy"] = hof_across_sow
    retreat_year = []
    df_hof_across_sow.to_csv("Inspect.csv")
    # print(df_hof_across_sow["Strategy"])
    for i in list(df_hof_across_sow["Strategy"]):
        try:
            ry = i.index(2)
        except:
            ry = 105
        retreat_year.append(ry)
    df_hof_across_sow["Retreat year"] = retreat_year
    df_hof_across_sow = df_hof_across_sow.replace([-np.inf], np.nan)
    df_hof_across_sow = df_hof_across_sow.replace([np.inf], np.nan)
    df_hof_across_sow = df_hof_across_sow.dropna()
    os.chdir(pwd)
    # Read in Pareto Front neglecting uncertainty
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "2objective_no_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    os.chdir(final_path)
    with open("hof_fitness.pkl", "rb") as handle:
        hof_fitness = pickle.load(handle)
    with open("hof.pkl", "rb") as handle:
        hof = pickle.load(handle)
    df_hof_best_guess_1 = pd.DataFrame(hof_fitness, columns=["Objectives", "Metrics"])
    df_hof_best_guess = pd.DataFrame(
        df_hof_best_guess_1["Objectives"].tolist(), columns=["Benefits", "Costs"]
    )
    df_hof_best_guess["Strategy"] = hof
    df_hof_best_guess.to_csv("Inspect.csv")
    retreat_year = []
    for i in df_hof_best_guess["Strategy"]:
        try:
            ry = i.index(2)
        except:
            ry = 105
        retreat_year.append(ry)
    df_hof_best_guess["Retreat year"] = retreat_year
    df_hof_best_guess = df_hof_best_guess.replace([-np.inf], np.nan)
    df_hof_best_guess = df_hof_best_guess.replace([np.inf], np.nan)
    df_hof_best_guess = df_hof_best_guess.dropna()
    os.chdir(pwd)
    # Modifications to values to ensure plotting consistency
    x = np.linspace(0, 7500000 * 1000 * 10**-9, 100000)
    y = [i * 2.5 for i in x]
    df_hof_best_guess["Costs"] *= 1000
    df_hof_best_guess["Benefits"] *= 1000
    df_hof_across_sow["Costs"] *= 1000
    df_hof_across_sow["Benefits"] *= 1000
    usace_costs_across_sow *= 1000
    usace_benefits_across_sow *= 1000
    max_npv_costs_across_sow *= 1000
    max_npv_benefits_across_sow *= 1000
    usace_costs *= 1000
    usace_benefits *= 1000
    max_npv_costs *= 1000
    max_npv_benefits *= 1000
    df_hof_across_sow["Costs"] /= 10**9
    df_hof_across_sow["Benefits"] /= 10**9
    df_hof_best_guess["Costs"] /= 10**9
    df_hof_best_guess["Benefits"] /= 10**9
    usace_costs_across_sow /= 10**9
    usace_benefits_across_sow /= 10**9
    max_npv_costs_across_sow /= 10**9
    max_npv_benefits_across_sow /= 10**9
    usace_costs /= 10**9
    usace_benefits /= 10**9
    max_npv_costs /= 10**9
    max_npv_benefits /= 10**9
    # Plotting here on
    golden_ratio = (1 + 5**0.5) / 2  # Approximately 1.618

    # Define the width of the figure
    width = 8  # You can choose any width you prefer

    # Calculate the height using the golden ratio
    height = width / golden_ratio
    fig = plt.figure(figsize=(2 * width, height))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cbar_ax = fig.add_subplot(gs[2])  # The colorbar will go here
    sc1 = create_plot(
        ax1,
        df_hof_best_guess["Costs"],
        df_hof_best_guess["Benefits"],
        df_hof_best_guess["Retreat year"],
        x,
        y,
        (0.2, 11.6),
        (max_npv_costs, max_npv_benefits),
        (usace_costs, usace_benefits),
        (2.5, 3.7),
        (4.5, 6),
        1,
    )
    sc2 = create_plot(
        ax2,
        df_hof_across_sow["Costs"],
        df_hof_across_sow["Benefits"],
        df_hof_across_sow["Retreat year"],
        x,
        y,
        (0.2, 11.6),
        (max_npv_costs_across_sow, max_npv_benefits_across_sow),
        (usace_costs_across_sow, usace_benefits_across_sow),
        (2.5, 4),
        (4.5, 6),
        0,
    )

    # Add the colorbar using the new axis
    cbar = fig.colorbar(sc2, cax=cbar_ax, extend="max")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_coords(0.5, -0.1)
    cbar.ax.set_xlabel("Retreat \n Year", fontsize=14, labelpad=5)
    ticks = cbar.get_ticks()
    new_ticks = list(ticks) + [cbar.vmax + 10]
    cbar.set_ticks(new_ticks)
    cbar.set_ticklabels([str(int(tick)) for tick in ticks] + ["No retreat"])
    cbar.ax.tick_params(labelsize=14)

    # Add panel labels
    ax1.set_title("Neglecting uncertainity", fontsize=16, fontweight="bold", pad=20)
    ax2.set_title("Considering uncertainity", fontsize=16, fontweight="bold", pad=20)
    ax1.text(
        -0.10,
        1.05,
        "A)",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.text(
        -0.10,
        1.05,
        "B)",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # Save the original directory
    original_dir = os.getcwd()
    # Go one folder up
    parent_dir = os.path.dirname(original_dir)
    os.chdir(parent_dir)
    figures_dir = os.path.join(parent_dir, "figures")
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # Go into "Figures" folder
    os.chdir(figures_dir)
    # Create "Main Figures" folder if it doesn't exist
    main_figures_dir = os.path.join(figures_dir, "main_figures")
    if not os.path.exists(main_figures_dir):
        os.mkdir(main_figures_dir)
    os.chdir(main_figures_dir)
    plt.savefig("5ObjectiveParetoFront.png", dpi=600, bbox_inches="tight")
    os.chdir(original_dir)
    return (npv_maximizing_strategy, usace_strategy)


def create_plot(
    ax,
    x_to_plot,
    y_to_plot,
    color_bar_scheme,
    x_fill,
    y_fill,
    heart_coords,
    npv_coords,
    usace_coords,
    npv_text_loc,
    usace_text_loc,
    label_names,
):
    cm = plt.cm.get_cmap("plasma")
    # cm = cc.cm['CET_CBL2']
    cm.set_over("deepskyblue")
    ax.fill_between(x_fill, y_fill, min(y_fill), color="lightgray", alpha=0.5, zorder=1)
    sc = ax.scatter(x_to_plot, y_to_plot, c=color_bar_scheme, vmin=0, vmax=100, cmap=cm)
    # Plot the green heart marker
    heart_x, heart_y = heart_coords  # Coordinates of the green heart
    npv_x, npv_y = npv_coords
    usace_x, usace_y = usace_coords
    ax.scatter(
        0.2,
        11.6,
        marker="$\u2764\ufe0f$",
        s=300,
        color="green",
        label="Ideal Point",
        zorder=5,
    )
    ax.scatter(npv_x, npv_y, color="#004d40", alpha=1, marker="P")
    ax.scatter(usace_x, usace_y, color="#0d47a1", alpha=1, marker="*")

    ax.annotate(
        "Utopian Point",
        xy=(heart_x, heart_y),  # The point to annotate
        xytext=(1.5, 10),  # The position of the text
        arrowprops=dict(color="green", arrowstyle="->"),  # Arrow properties
        fontsize=14,
        color="green",
        zorder=4,
    )
    # Annotate another point with "NPV maximizing strategy" in rebeccapurple
    npv_x, npv_y = npv_coords  # Coordinates of the NPV maximizing strategy
    ax.annotate(
        "NPV maximizing\nstrategy",  # Newline added
        xy=(npv_x, npv_y),  # The point to annotate
        xytext=npv_text_loc,  # Custom text position
        arrowprops=dict(arrowstyle="->", color="#004d40"),  # Arrow properties
        fontsize=14,
        color="#004d40",
        zorder=4,
        linespacing=1.2,
    )  # Adjust linespacing

    # Plot the pink scatter point for "USACE strategy w/o retreat"
    # ax.plot(usace_x, usace_y, '*', color='rebeccapurple', markersize=10, label="USACE strategy w/o retreat", zorder=3)
    ax.annotate(
        "USACE strategy\nw/o retreat",  # Newline added
        xy=(usace_x, usace_y),  # The point to annotate
        xytext=usace_text_loc,  # Custom text position
        arrowprops=dict(arrowstyle="->", color="steelblue"),  # Arrow properties
        fontsize=14,
        color="steelblue",
        zorder=4,
        linespacing=1.2,
    )  # Adjust linespacing
    if label_names == 0:
        ax.set_xlabel("Total Expected\n Discounted Costs ($B)", fontsize=14)
        ax.set_ylabel("Total Expected\n Discounted Benefits ($B)", fontsize=14)
    else:
        ax.set_xlabel("Total \n Discounted Costs ($B)", fontsize=14)
        ax.set_ylabel("Total \n Discounted Benefits ($B)", fontsize=14)
    ax.text(3, 1, "Fails benefit cost\nratio threshold test", fontsize=14, color="gray")
    ax.plot(
        x_fill,
        y_fill,
        label="USACE funding \n threshold test",
        color="steelblue",
        linestyle="--",
    )
    # Adjust the x-axis and y-axis limits with some padding
    ax.set_xlim(
        0, 7.5
    )  # Set limits to ensure all points and lines are within the visible range
    ax.set_ylim(0, 12)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["left"].set_position("zero")
    return sc


def billions(x, pos):
    return "%1.1fB" % (x * 1e-9)


def generate_multi_axis_parallel_plots_with_satisficing_strategies(baseline_directory):
    pwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "5objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    os.chdir(final_path)
    with open("hof_fitness_across_sow.pkl", "rb") as handle:
        hof_fitness_across_sow = pickle.load(handle)
    with open("hof_across_sow.pkl", "rb") as handle:
        hof_across_sow = pickle.load(handle)
    df_hof_fitness = pd.DataFrame(
        hof_fitness_across_sow,
        columns=["NPV", "Benefits", "Investment Costs", "Damage Costs", "Reliability"],
    )

    retreat_years = []
    strategy = []
    for individual in hof_across_sow:
        if 2 in individual:
            retreat_year = individual.index(2)
        else:
            retreat_year = 105  # Set to 105 if no '2' is present
        retreat_years.append(retreat_year)
        strategy.append(individual)

    # Add the Retreat Year as a new column in the DataFrame
    df_hof_fitness["Retreat Year"] = retreat_years
    df_hof_fitness["strategy"] = strategy
    df_hof_fitness.to_csv("Inspect.csv")
    # df_no_retreat.to_csv("take_a_look.csv")
    # df_hof_fitness = df_hof_fitness.dropna()
    df_hof_fitness = df_hof_fitness.replace([-np.inf], np.nan)
    df_hof_fitness = df_hof_fitness.dropna()
    df_no_retreat = df_hof_fitness[df_hof_fitness["Retreat Year"] > 100]
    df_hof_fitness["NPV"] = df_hof_fitness["NPV"] / 10**6
    df_hof_fitness["Benefits"] = df_hof_fitness["Benefits"] / 10**6
    df_hof_fitness["Damage Costs"] = df_hof_fitness["Damage Costs"] / 10**6
    df_hof_fitness["Investment Costs"] = df_hof_fitness["Investment Costs"] / 10**6
    # Filter data for the second, third, and fourth figures
    df_hof_no_retreat = df_hof_fitness[df_hof_fitness["Retreat Year"] > 100]
    df_hof_bcr_2_5 = df_hof_fitness[
        df_hof_fitness["Benefits"]
        / (df_hof_fitness["Investment Costs"] + df_hof_fitness["Damage Costs"])
        > 2.5
    ]
    df_hof_reliability_90 = df_hof_fitness[df_hof_fitness["Reliability"] > 0.90]
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 0.12], hspace=0.2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    cbar_ax = fig.add_subplot(gs[5])
    objs_reorg_1, tops_1, bottoms_1 = reorganize_objs(
        df_hof_fitness,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        ideal_direction="top",
        minmaxs=["max", "max", "min", "min", "max", "max"],
    )

    # Plot 1: df_hof_fitness
    custom_parallel_coordinates(
        df_hof_fitness,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        axis_labels=[
            "Net Present \n Value ($B)",
            "Total \n Discounted \n Benefits ($B)",
            "Total Discounted \n Investment \n Costs ($B)",
            "Total Discounted \n Damage \n Costs ($B)",
            "Reliability",
            "Retreat Year",
        ],
        minmaxs=["max", "max", "min", "min", "max", "max"],
        color_by_continuous=5,
        color_palette_continuous="tab20b",
        zorder_by=0,
        ideal_direction="top",
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=12,
        figsize=(20, 8),
        ax=ax1,  # Pass the first subplot's axes
    )
    first_plot_ylim = ax1.get_ylim()

    objs_reorg_2, tops_2, bottoms_2 = reorganize_objs(
        df_hof_no_retreat,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        ideal_direction="top",
        minmaxs=["max", "max", "max", "max", "max", "max"],
    )
    # Plot 2: df_hof_no_retreat (Retreat Year > 105)
    custom_parallel_coordinates(
        df_hof_no_retreat,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        axis_labels=[
            "Net Present \n Value ($B)",
            "Total \n Discounted \n Benefits ($B)",
            "Total Discounted \n Investment \n Costs ($B)",
            "Total Discounted \n Damage \n Costs ($B)",
            "Reliability",
            "Retreat Year",
        ],
        minmaxs=["max", "max", "max", "max", "max", "max"],
        color_by_continuous=5,
        color_palette_continuous="tab20b",
        zorder_by=0,
        ideal_direction="top",
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=12,
        figsize=(20, 8),
        ax=ax2,
        tops=tops_1,
        bottoms=bottoms_1,
    )
    objs_reorg_3, tops_3, bottoms_3 = reorganize_objs(
        df_hof_bcr_2_5,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        ideal_direction="top",
        minmaxs=["max", "max", "max", "max", "max", "max"],
    )

    # Plot 3: df_hof_bcr_2.5 (BCR > 2.5)
    custom_parallel_coordinates(
        df_hof_bcr_2_5,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        axis_labels=[
            "Net Present \n Value ($B)",
            "Total \n Discounted \n Benefits ($B)",
            "Total Discounted \n Investment \n Costs ($B)",
            "Total Discounted \n Damage \n Costs ($B)",
            "Reliability",
            "Retreat Year",
        ],
        minmaxs=["max", "max", "max", "max", "max", "max"],
        color_by_continuous=5,
        color_palette_continuous="tab20b",
        zorder_by=0,
        ideal_direction="top",
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=12,
        figsize=(20, 8),
        ax=ax3,
        tops=tops_1,
        bottoms=bottoms_1,
    )
    objs_reorg_4, tops_4, bottoms_4 = reorganize_objs(
        df_hof_reliability_90,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        ideal_direction="top",
        minmaxs=["max", "max", "max", "max", "max", "max"],
    )

    # Plot 4: df_hof_reliability_90 (Reliability > 90%)
    custom_parallel_coordinates(
        df_hof_reliability_90,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        axis_labels=[
            "Net Present \n Value ($B)",
            "Total \n Discounted \n Benefits ($B)",
            "Total Discounted \n Investment \n Costs ($B)",
            "Total Discounted \n Damage \n Costs ($B)",
            "Reliability",
            "Retreat Year",
        ],
        minmaxs=["max", "max", "max", "max", "max", "max"],
        color_by_continuous=5,
        color_palette_continuous="tab20b",
        zorder_by=0,
        ideal_direction="top",
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=12,
        figsize=(20, 8),
        ax=ax4,
        tops=tops_1,
        bottoms=bottoms_1,  # Pass the fourth subplot's axes
    )

    columns_axes = [
        "NPV",
        "Benefits",
        "Investment Costs",
        "Damage Costs",
        "Reliability",
        "Retreat Year",
    ]
    axis_labels = [
        "Net Present \n Value ($B)",
        "Total \n Discounted \n Benefits ($B)",
        "Total Discounted \n Investment \n Costs ($B)",
        "Total Discounted \n Damage \n Costs ($B)",
        "Reliability",
        "Retreat Year",
    ]
    # empty_df = pd.DataFrame(columns=columns_axes)
    empty_df = df_hof_fitness[
        (
            df_hof_fitness["Benefits"]
            / (df_hof_fitness["Investment Costs"] + df_hof_fitness["Damage Costs"])
            > 2.5
        )
        & (df_hof_fitness["Reliability"] > 0.90)
    ]
    objs_reorg_5, tops_5, bottoms_5 = reorganize_objs(
        empty_df,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        ideal_direction="top",
        minmaxs=["max", "max", "max", "max", "max", "max"],
    )

    custom_parallel_coordinates(
        empty_df,
        columns_axes=[
            "NPV",
            "Benefits",
            "Investment Costs",
            "Damage Costs",
            "Reliability",
            "Retreat Year",
        ],
        axis_labels=[
            "Net Present \n Value ($B)",
            "Total \n Discounted \n Benefits ($B)",
            "Total Discounted \n Investment \n Costs ($B)",
            "Total Discounted \n Damage \n Costs ($B)",
            "Reliability",
            "Retreat Year",
        ],
        minmaxs=["max", "max", "max", "max", "max", "max"],
        color_by_continuous=5,
        color_palette_continuous="tab20b",
        zorder_by=0,
        ideal_direction="top",
        alpha_base=0.8,
        lw_base=1.5,
        fontsize=12,
        figsize=(20, 8),
        ax=ax5,
        tops=tops_1,
        bottoms=bottoms_1,  # Pass the fourth subplot's axes
    )
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel(ax.get_xlabel(), labelpad=10)  # Increase x-axis label padding
        ax.set_ylabel(ax.get_ylabel(), labelpad=10)  # Increase y-axis label padding
    # Set up the colorbar axis
    # cbar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02])  # [left, bottom, width, height] for colorbar
    # cbar_ax = fig.add_subplot(gs[5])

    # Set up the colormap and normalization
    cmap = cm.get_cmap("tab20b")
    cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))
    cmap.set_over("deepskyblue")  # Set the color for values > 100

    # Normalize retreat year values (assuming retreat year is the 6th column)
    norm = mcolors.Normalize(vmin=df_hof_fitness[columns_axes[5]].min(), vmax=100)

    # Create ScalarMappable for color mapping
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create the colorbar below all subplots
    cb = plt.colorbar(
        mappable,
        cax=cbar_ax,
        orientation="horizontal",
        shrink=0.5,
        label=columns_axes[5],
        extend="max",
        alpha=0.8,
    )

    # Set colorbar ticks
    ticks = cb.get_ticks()  # Get the default ticks
    new_ticks = list(ticks) + [cb.vmax + 10]  # Add 'No Retreat' tick (vmax + 10)
    cb.set_ticks(new_ticks)  # Set the modified ticks
    cb.set_ticklabels(
        [str(int(tick)) for tick in ticks] + ["No\nretreat"]
    )  # Format tick labels

    # Set label and label position
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_coords(0.5, -1.3)
    cb.ax.set_xlabel("Retreat Year", fontsize=14)
    cb.ax.tick_params(labelsize=14)

    # Make sure to apply the correct label again
    # cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=14, labelpad=8)

    ax1.text(
        -0.02,
        1.02,
        "A)",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.text(
        -0.02,
        1.02,
        "B)",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax3.text(
        -0.02,
        1.02,
        "C)",
        transform=ax3.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax4.text(
        -0.02,
        1.02,
        "D)",
        transform=ax4.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax5.text(
        -0.02,
        1.02,
        "E)",
        transform=ax5.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax1.text(
        -0.02,
        0.75,
        "Full\nPareto\nFront",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax1.text(
        -0.02,
        0.75,
        "Strategies\nw/o\nRetreat",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax1.text(
        -0.02,
        0.75,
        "Strategies\nwith\nBCR>2.5",
        transform=ax3.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax1.text(
        -0.02,
        0.75,
        "Strategies\nwith\nReliability>90%",
        transform=ax4.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax1.text(
        -0.02,
        0.75,
        "Strategies\nwith\nBCR>2.5\nand\nReliability>90%",
        transform=ax5.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    os.chdir(pwd)
    original_dir = os.getcwd()
    # Go one folder up
    parent_dir = os.path.dirname(original_dir)
    os.chdir(parent_dir)
    figures_dir = os.path.join(parent_dir, "figures")
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # Go into "Figures" folder
    os.chdir(figures_dir)
    # Create "Main Figures" folder if it doesn't exist
    main_figures_dir = os.path.join(figures_dir, "main_figures")
    if not os.path.exists(main_figures_dir):
        os.mkdir(main_figures_dir)
    os.chdir(main_figures_dir)
    empty_df.to_csv("Satisficing_df.csv")
    # Save the final figure
    plt.savefig(
        "5_panel_parallel_coordinates.png",
        dpi=600,
        bbox_inches="tight",
    )
    os.chdir(original_dir)
    return "5 Objectives Done!"


def custom_parallel_coordinates(
    objs,
    columns_axes=None,
    axis_labels=None,
    ideal_direction="top",
    minmaxs=None,
    color_by_continuous=None,
    color_palette_continuous=None,
    color_by_categorical=None,
    color_palette_categorical=None,
    colorbar_ticks_continuous=None,
    color_dict_categorical=None,
    zorder_by=None,
    zorder_num_classes=10,
    zorder_direction="ascending",
    alpha_base=0.8,
    brushing_dict=None,
    alpha_brush=0.05,
    lw_base=1.5,
    fontsize=14,
    figsize=(11, 6),
    save_fig_filename=None,
    ax=None,
    tops=None,
    bottoms=None,
):
    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, gridspec_kw={"hspace": 0.1, "wspace": 0.1}
        )

    ### reorganize & normalize objective data
    # objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)
    if tops is None or bottoms is None:
        objs_reorg, tops, bottoms = reorganize_objs(
            objs, columns_axes, ideal_direction, minmaxs
        )
    else:
        objs_reorg = objs[columns_axes]
        # Normalize data manually if tops and bottoms are provided
        for i, minmax in enumerate(minmaxs):
            if minmax == "max":
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i] - bottoms[i]) / (
                    tops[i] - bottoms[i]
                )
            else:
                objs_reorg.iloc[:, i] = (tops[i] - objs_reorg.iloc[:, i]) / (
                    tops[i] - bottoms[i]
                )

    ### apply any brushing criteria
    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.0
        ### iteratively apply all brushing criteria to get satisficing set of solutions
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == "<":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == "<=":
                satisfice = np.logical_and(
                    satisfice, objs.iloc[:, col_idx] <= threshold
                )
            elif operator == ">":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == ">=":
                satisfice = np.logical_and(
                    satisfice, objs.iloc[:, col_idx] >= threshold
                )

            ### add rectangle patch to plot to represent brushing
            threshold_norm = (threshold - bottoms[col_idx]) / (
                tops[col_idx] - bottoms[col_idx]
            )
            if ideal_direction == "top" and minmaxs[col_idx] == "max":
                if operator in ["<", "<="]:
                    rect = Rectangle(
                        [col_idx - 0.05, threshold_norm], 0.1, 1 - threshold_norm
                    )
                elif operator in [">", ">="]:
                    rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            elif ideal_direction == "top" and minmaxs[col_idx] == "min":
                if operator in ["<", "<="]:
                    rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
                elif operator in [">", ">="]:
                    rect = Rectangle(
                        [col_idx - 0.05, threshold_norm], 0.1, 1 - threshold_norm
                    )
            if ideal_direction == "bottom" and minmaxs[col_idx] == "max":
                if operator in ["<", "<="]:
                    rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
                elif operator in [">", ">="]:
                    rect = Rectangle(
                        [col_idx - 0.05, threshold_norm], 0.1, 1 - threshold_norm
                    )
            elif ideal_direction == "bottom" and minmaxs[col_idx] == "min":
                if operator in ["<", "<="]:
                    rect = Rectangle(
                        [col_idx - 0.05, threshold_norm], 0.1, 1 - threshold_norm
                    )
                elif operator in [">", ">="]:
                    rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)

            pc = PatchCollection([rect], facecolor="grey", alpha=0.5, zorder=3)
            ax.add_collection(pc)

    ### loop over all solutions/rows & plot on parallel axis plot
    for i in range(objs_reorg.shape[0]):
        if color_by_continuous is not None:
            # value = objs_reorg[columns_axes[color_by_continuous]].iloc[i]
            # color = get_color(value,
            #                   color_by_continuous, color_palette_continuous,
            #                   color_by_categorical, color_dict_categorical)
            cmap = cm.get_cmap("tab20b")
            norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize to 0-1 range
            value = objs_reorg[columns_axes[color_by_continuous]].iloc[i]
            color = get_color(value, norm, cmap)
            # print("value", value)
            # if value>0.99:
            #     print(value,color)
        elif color_by_categorical is not None:
            color = get_color(
                objs[color_by_categorical].iloc[i],
                color_by_continuous,
                color_palette_continuous,
                color_by_categorical,
                color_dict_categorical,
            )

        ### order lines according to ascending or descending values of one of the objectives?
        if zorder_by is None:
            zorder = 4
        else:
            zorder = get_zorder(
                objs_reorg[columns_axes[zorder_by]].iloc[i],
                zorder_num_classes,
                zorder_direction,
            )

        ### apply any brushing?
        if brushing_dict is not None:
            if satisfice.iloc[i]:
                alpha = alpha_base
                lw = lw_base
            else:
                alpha = alpha_brush
                lw = 1
                zorder = 2
        else:
            alpha = alpha_base
            lw = lw_base

        ### loop over objective/column pairs & plot lines between parallel axes
        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    ### Add top/bottom ranges (same as before)
    ### add top/bottom ranges
    for j in range(len(columns_axes)):
        ax.annotate(
            str(round(tops[j], 2)),
            [j, 1.02],
            ha="center",
            va="bottom",
            zorder=5,
            fontsize=fontsize,
        )

        if j == len(columns_axes) - 1:
            if columns_axes[j] == "Reliability":
                ax.annotate(
                    str(round(bottoms[j], 2)),
                    [j, -0.02],
                    ha="center",
                    va="top",
                    zorder=5,
                    fontsize=fontsize,
                )
            else:
                ax.annotate(
                    str(round(bottoms[j], 2)) + "+",
                    [j, -0.02],
                    ha="center",
                    va="top",
                    zorder=5,
                    fontsize=fontsize,
                )
        else:
            if columns_axes[j] == "Reliability":
                ax.annotate(
                    str(round(bottoms[j], 2)),
                    [j, -0.02],
                    ha="center",
                    va="top",
                    zorder=5,
                    fontsize=fontsize,
                )
            else:
                ax.annotate(
                    str(round(bottoms[j], 2)),
                    [j, -0.02],
                    ha="center",
                    va="top",
                    zorder=5,
                    fontsize=fontsize,
                )

        ax.plot([j, j], [0, 1], c="k", zorder=1)

    ### other aesthetics
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    if ideal_direction == "top":
        ax.arrow(
            -0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color="k", lw=1.5
        )
    elif ideal_direction == "bottom":
        ax.arrow(
            -0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color="k", lw=1.5
        )
    ax.annotate(
        "Direction of\npreference",
        xy=(-0.3, 0.5),
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
    )

    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(-0.4, 1.1)

    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha="center", va="top", fontsize=fontsize)
    ax.patch.set_alpha(0)

    ### colorbar for continuous legend
    if color_by_continuous is not None:
        print("Yes!")
        # cmap = cm.get_cmap(color_palette_continuous)
        # cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))
        # cmap.set_over('deepskyblue')  # Set the color for values > 100
        # norm = mcolors.Normalize(vmin=objs[columns_axes[color_by_continuous]].min(), vmax=100)
        # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        # # mappable.set_over('deepskyblue')
        # cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4,
        #                   label=axis_labels[color_by_continuous], pad=0.03, extend ='max',
        #                   alpha=0.8)
        # # cb.set_over('deepskyblue')
        # if colorbar_ticks_continuous is not None:
        #     _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
        #                          fontsize=fontsize)
        # # cb.ax.xaxis.set_label_position('bottom')
        # # cb.ax.xaxis.set_ticks_position('bottom')
        # # cb.ax.xaxis.set_label_coords(0.5, -0.1)
        # # cb.ax.set_xlabel('Retreat Year', fontsize=fontsize, labelpad=5)
        # ticks = cb.get_ticks()
        # new_ticks = list(ticks) + [cb.vmax + 7]
        # cb.set_ticks(new_ticks)
        # cb.set_ticklabels([str(int(tick)) for tick in ticks] + ['No\nretreat'])
        # _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize, labelpad=8)

    ### categorical legend
    elif color_by_categorical is not None:
        leg = []
        for label, color in color_dict_categorical.items():
            leg.append(
                Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label)
            )
        _ = ax.legend(
            handles=leg,
            loc="lower center",
            ncol=max(3, len(color_dict_categorical)),
            bbox_to_anchor=[0.5, -0.07],
            frameon=False,
            fontsize=fontsize,
        )

    ### Save figure if filename is provided
    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches="tight", dpi=300)


def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    ### if min/max directions not given for each axis, assume all should be maximized
    if minmaxs is None:
        minmaxs = ["max"] * len(columns_axes)

    ### get subset of dataframe columns that will be shown as parallel axes
    objs_reorg = objs[columns_axes]

    ### reorganize & normalize data to go from 0 (bottom of figure) to 1 (top of figure),
    ### based on direction of preference for figure and individual axes
    if ideal_direction == "bottom":
        tops = objs_reorg.min(axis=0)
        bottoms = objs_reorg.max(axis=0)
        for i, minmax in enumerate(minmaxs):
            if minmax == "max":
                objs_reorg.iloc[:, i] = (
                    objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]
                ) / (
                    objs_reorg.iloc[:, i].max(axis=0)
                    - objs_reorg.iloc[:, i].min(axis=0)
                )
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, -1] = (
                    objs_reorg.iloc[:, -1] - objs_reorg.iloc[:, -1].min(axis=0)
                ) / (
                    objs_reorg.iloc[:, -1].max(axis=0)
                    - objs_reorg.iloc[:, -1].min(axis=0)
                )
    elif ideal_direction == "top":
        tops = objs_reorg.max(axis=0)
        bottoms = objs_reorg.min(axis=0)
        for i, minmax in enumerate(minmaxs):
            if minmax == "max":
                objs_reorg.iloc[:, i] = (
                    objs_reorg.iloc[:, i] - objs_reorg.iloc[:, i].min(axis=0)
                ) / (
                    objs_reorg.iloc[:, i].max(axis=0)
                    - objs_reorg.iloc[:, i].min(axis=0)
                )
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (
                    objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]
                ) / (
                    objs_reorg.iloc[:, i].max(axis=0)
                    - objs_reorg.iloc[:, i].min(axis=0)
                )

    return objs_reorg, tops, bottoms


def get_color(value, norm, cmap, threshold=0.99):
    if value >= threshold:  # Check if the normalized value is close to 1
        return "deepskyblue"
    else:
        return cmap(
            norm(value)
        )  # Use ScalarMappable to correctly map the value to the colormap


### function to get zorder value for ordering lines on plot.
### This works by binning a given axis' values and mapping to discrete classes.
def get_zorder(norm_value, zorder_num_classes, zorder_direction):
    xgrid = np.arange(0, 1.001, 1 / zorder_num_classes)
    if zorder_direction == "ascending":
        return 4 + np.sum(norm_value > xgrid)
    elif zorder_direction == "descending":
        return 4 + np.sum(norm_value < xgrid)


def identify_satisfycing_strategies(baseline_directory):
    pwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "5objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    os.chdir(final_path)
    with open("hof_fitness_across_sow.pkl", "rb") as handle:
        hof_fitness_across_sow = pickle.load(handle)
    with open("hof_across_sow.pkl", "rb") as handle:
        hof_across_sow = pickle.load(handle)
    df_hof_fitness = pd.DataFrame(
        hof_fitness_across_sow,
        columns=["NPV", "Benefits", "Investment Costs", "Damage Costs", "Reliability"],
    )

    retreat_years = []
    strategy = []
    for individual in hof_across_sow:
        if 2 in individual:
            retreat_year = individual.index(2)
        else:
            retreat_year = 105  # Set to 105 if no '2' is present
        retreat_years.append(retreat_year)
        strategy.append(individual)

    # Add the Retreat Year as a new column in the DataFrame
    df_hof_fitness["Retreat Year"] = retreat_years
    df_hof_fitness["strategy"] = strategy
    df_hof_fitness.to_csv("Inspect.csv")
    # df_no_retreat.to_csv("take_a_look.csv")
    # df_hof_fitness = df_hof_fitness.dropna()
    df_hof_fitness = df_hof_fitness.replace([-np.inf], np.nan)
    df_hof_fitness = df_hof_fitness.dropna()
    df_no_retreat = df_hof_fitness[df_hof_fitness["Retreat Year"] > 100]
    df_hof_fitness["NPV"] = df_hof_fitness["NPV"] / 10**6
    df_hof_fitness["Benefits"] = df_hof_fitness["Benefits"] / 10**6
    df_hof_fitness["Damage Costs"] = df_hof_fitness["Damage Costs"] / 10**6
    df_hof_fitness["Investment Costs"] = df_hof_fitness["Investment Costs"] / 10**6
    # Filter data for the second, third, and fourth figures
    df_hof_no_retreat = df_hof_fitness[df_hof_fitness["Retreat Year"] > 100]
    df_hof_bcr_2_5 = df_hof_fitness[
        df_hof_fitness["Benefits"]
        / (df_hof_fitness["Investment Costs"] + df_hof_fitness["Damage Costs"])
        > 2.5
    ]
    df_hof_reliability_90 = df_hof_fitness[df_hof_fitness["Reliability"] > 0.90]
    # empty_df = pd.DataFrame(columns=columns_axes)
    empty_df = df_hof_fitness[
        (
            df_hof_fitness["Benefits"]
            / (df_hof_fitness["Investment Costs"] + df_hof_fitness["Damage Costs"])
            > 2.5
        )
        & (df_hof_fitness["Reliability"] > 0.90)
    ]
    max_npv = empty_df["NPV"].max()
    robustness_satisficing_strategy = list(
        empty_df.loc[empty_df["NPV"] == max_npv, "strategy"]
    )[0]
    print(robustness_satisficing_strategy)
    os.chdir(pwd)
    # Read in pareto front considering uncertainty
    pwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_dir = os.path.join(parent_dir, "results_data", "baseline_optimization_runs")
    # Define the folder to check/create
    final_folder = "2objective_considering_uncertainty"
    final_path = os.path.join(target_dir, final_folder)
    os.chdir(final_path)
    with open("hof_fitness_across_sow.pkl", "rb") as handle:
        hof_fitness_across_sow_2Objectives = pickle.load(handle)
    with open("hof_across_sow.pkl", "rb") as handle:
        hof_across_sow_2Objectives = pickle.load(handle)
    df_hof_across_sow_2Objectives = pd.DataFrame(
        hof_fitness_across_sow_2Objectives, columns=["Benefits", "Costs"]
    )
    df_hof_across_sow_2Objectives["Strategy"] = hof_across_sow_2Objectives
    retreat_year = []
    df_hof_across_sow_2Objectives.to_csv("Inspect.csv")
    # print(df_hof_across_sow["Strategy"])
    for i in list(df_hof_across_sow_2Objectives["Strategy"]):
        try:
            ry = i.index(2)
        except:
            ry = 105
        retreat_year.append(ry)
    df_hof_across_sow_2Objectives["Retreat year"] = retreat_year
    df_hof_across_sow_2Objectives = df_hof_across_sow_2Objectives.replace(
        [-np.inf], np.nan
    )
    df_hof_across_sow_2Objectives = df_hof_across_sow_2Objectives.replace(
        [np.inf], np.nan
    )
    df_hof_across_sow_2Objectives = df_hof_across_sow_2Objectives.dropna()
    df_hof_across_sow_2Objectives["BCR"] = (
        df_hof_across_sow_2Objectives["Benefits"]
        / df_hof_across_sow_2Objectives["Costs"]
    )
    df_hof_across_sow_2Objectives["NPV"] = (
        df_hof_across_sow_2Objectives["Benefits"]
        - df_hof_across_sow_2Objectives["Costs"]
    )
    filtered_df = df_hof_across_sow_2Objectives[
        df_hof_across_sow_2Objectives["BCR"] > 2.5
    ]
    # Find index of row with maximum NPV in the filtered DataFrame
    max_npv_index = filtered_df["NPV"].idxmax()
    # Extract the corresponding strategy
    threshold_satisficing_strategy = filtered_df.loc[max_npv_index, "Strategy"]
    os.chdir(pwd)
    return threshold_satisficing_strategy, robustness_satisficing_strategy


def grouped_radial(
    SAresults,
    parameters,
    parameter_names,
    ax,
    radSc=2.0,
    scaling=1,
    widthSc=0.5,
    STthick=1,
    varNameMult=1.45,
    colors=None,
    groups=None,
    gpNameMult=1.7,
    threshold="conf",
):
    # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
    # fig, ax = plt.subplots(1, 1)
    color_map = {}

    # initialize parameters and colors
    if groups is None:
        if colors is None:
            colors = ["k"]

        for i, parameter in enumerate(parameters):
            color_map[parameter] = colors[i % len(colors)]
    else:
        if colors is None:
            colors = sns.color_palette("deep", max(3, len(groups)))

        for i, key in enumerate(groups.keys()):
            # parameters.extend(groups[key])

            for parameter in groups[key]:
                color_map[parameter] = colors[i % len(colors)]

    n = len(parameters)
    angles = radSc * math.pi * np.arange(0, n) / n
    x = radSc * np.cos(angles)
    y = radSc * np.sin(angles)

    # plot second-order indices
    for i, j in itertools.combinations(range(n), 2):
        # key1 = parameters[i]
        # key2 = parameters[j]

        if is_significant(SAresults["S2"][i][j], SAresults["S2_conf"][i][j], threshold):
            angle = math.atan((y[j] - y[i]) / (x[j] - x[i]))

            if y[j] - y[i] < 0:
                angle += math.pi

            line_hw = scaling * (max(0, SAresults["S2"][i][j]) ** widthSc) / 2

            coords = np.empty((4, 2))
            coords[0, 0] = x[i] - line_hw * math.sin(angle)
            coords[1, 0] = x[i] + line_hw * math.sin(angle)
            coords[2, 0] = x[j] + line_hw * math.sin(angle)
            coords[3, 0] = x[j] - line_hw * math.sin(angle)
            coords[0, 1] = y[i] + line_hw * math.cos(angle)
            coords[1, 1] = y[i] - line_hw * math.cos(angle)
            coords[2, 1] = y[j] - line_hw * math.cos(angle)
            coords[3, 1] = y[j] + line_hw * math.cos(angle)

            ax.add_artist(plt.Polygon(coords, color="0.75"))

    # plot total order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults["ST"][i], SAresults["ST_conf"][i], threshold):
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["ST"][i] ** widthSc) / 2,
                    color="w",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["ST"][i] ** widthSc) / 2,
                    lw=STthick,
                    color="0.4",
                    fill=False,
                )
            )

    # plot first-order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults["S1"][i], SAresults["S1_conf"][i], threshold):
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["S1"][i] ** widthSc) / 2,
                    color="0.4",
                )
            )

    # Add labels for parameters (rotation set to 0 for all labels)
    for i, key in enumerate(parameter_names):
        param_short_form = parameters[i]
        ax.text(
            varNameMult * x[i],
            varNameMult * y[i],
            key,
            ha="center",
            va="center",
            rotation=0,  # Set rotation to 0 for horizontal text
            color=color_map[param_short_form],
            fontsize=23,
            fontweight="bold",
        )

    # Add labels for groups (rotation also set to 0)
    if groups is not None:
        for i, group in enumerate(groups.keys()):
            if group == "Socio-economic":
                ax.text(
                    -3.0,
                    -3.3,
                    "Socio-\neconomic",
                    ha="center",
                    va="center",
                    rotation=0,
                    color=colors[i % len(colors)],
                    fontsize=34,
                    fontweight="bold",
                )
            elif group == "Geo-physical":
                ax.text(
                    3.1,
                    3.3,
                    "Geo-\nphysical",
                    ha="center",
                    va="center",
                    rotation=0,
                    color=colors[i % len(colors)],
                    fontsize=34,
                    fontweight="bold",
                )
            else:
                group_angle = np.mean(
                    [angles[j] for j in range(n) if parameters[j] in groups[group]]
                )
                ax.text(
                    gpNameMult * radSc * math.cos(group_angle),
                    gpNameMult * radSc * math.sin(group_angle),
                    group,
                    ha="center",
                    va="center",
                    rotation=0,  # Set rotation to 0 for group labels
                    color=colors[i % len(colors)],
                    fontsize=24,
                    fontweight="bold",
                )

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.set_xlim([-2 * radSc, 2 * radSc])
    ax.set_ylim([-2 * radSc, 2 * radSc])
    return fig


def grouped_radial(
    SAresults,
    parameters,
    parameter_names,
    ax,
    radSc=2.0,
    scaling=1,
    widthSc=0.5,
    STthick=1,
    varNameMult=1.45,
    colors=None,
    groups=None,
    gpNameMult=1.7,
    threshold="conf",
):
    # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
    # fig, ax = plt.subplots(1, 1)
    color_map = {}

    # initialize parameters and colors
    if groups is None:
        if colors is None:
            colors = ["k"]

        for i, parameter in enumerate(parameters):
            color_map[parameter] = colors[i % len(colors)]
    else:
        if colors is None:
            colors = sns.color_palette("deep", max(3, len(groups)))

        for i, key in enumerate(groups.keys()):
            # parameters.extend(groups[key])

            for parameter in groups[key]:
                color_map[parameter] = colors[i % len(colors)]

    n = len(parameters)
    angles = radSc * math.pi * np.arange(0, n) / n
    x = radSc * np.cos(angles)
    y = radSc * np.sin(angles)

    # plot second-order indices
    for i, j in itertools.combinations(range(n), 2):
        # key1 = parameters[i]
        # key2 = parameters[j]

        if is_significant(SAresults["S2"][i][j], SAresults["S2_conf"][i][j], threshold):
            angle = math.atan((y[j] - y[i]) / (x[j] - x[i]))

            if y[j] - y[i] < 0:
                angle += math.pi

            line_hw = scaling * (max(0, SAresults["S2"][i][j]) ** widthSc) / 2

            coords = np.empty((4, 2))
            coords[0, 0] = x[i] - line_hw * math.sin(angle)
            coords[1, 0] = x[i] + line_hw * math.sin(angle)
            coords[2, 0] = x[j] + line_hw * math.sin(angle)
            coords[3, 0] = x[j] - line_hw * math.sin(angle)
            coords[0, 1] = y[i] + line_hw * math.cos(angle)
            coords[1, 1] = y[i] - line_hw * math.cos(angle)
            coords[2, 1] = y[j] - line_hw * math.cos(angle)
            coords[3, 1] = y[j] + line_hw * math.cos(angle)

            ax.add_artist(plt.Polygon(coords, color="0.75"))

    # plot total order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults["ST"][i], SAresults["ST_conf"][i], threshold):
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["ST"][i] ** widthSc) / 2,
                    color="w",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["ST"][i] ** widthSc) / 2,
                    lw=STthick,
                    color="0.4",
                    fill=False,
                )
            )

    # plot first-order indices
    for i, key in enumerate(parameters):
        if is_significant(SAresults["S1"][i], SAresults["S1_conf"][i], threshold):
            ax.add_artist(
                plt.Circle(
                    (x[i], y[i]),
                    scaling * (SAresults["S1"][i] ** widthSc) / 2,
                    color="0.4",
                )
            )

    # Add labels for parameters (rotation set to 0 for all labels)
    for i, key in enumerate(parameter_names):
        param_short_form = parameters[i]
        ax.text(
            varNameMult * x[i],
            varNameMult * y[i],
            key,
            ha="center",
            va="center",
            rotation=0,  # Set rotation to 0 for horizontal text
            color=color_map[param_short_form],
            fontsize=23,
            fontweight="bold",
        )

    # Add labels for groups (rotation also set to 0)
    if groups is not None:
        for i, group in enumerate(groups.keys()):
            if group == "Socio-economic":
                ax.text(
                    -3.0,
                    -3.3,
                    "Socio-\neconomic",
                    ha="center",
                    va="center",
                    rotation=0,
                    color=colors[i % len(colors)],
                    fontsize=34,
                    fontweight="bold",
                )
            elif group == "Geo-physical":
                ax.text(
                    3.1,
                    3.3,
                    "Geo-\nphysical",
                    ha="center",
                    va="center",
                    rotation=0,
                    color=colors[i % len(colors)],
                    fontsize=34,
                    fontweight="bold",
                )
            else:
                group_angle = np.mean(
                    [angles[j] for j in range(n) if parameters[j] in groups[group]]
                )
                ax.text(
                    gpNameMult * radSc * math.cos(group_angle),
                    gpNameMult * radSc * math.sin(group_angle),
                    group,
                    ha="center",
                    va="center",
                    rotation=0,  # Set rotation to 0 for group labels
                    color=colors[i % len(colors)],
                    fontsize=24,
                    fontweight="bold",
                )

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.set_xlim([-2 * radSc, 2 * radSc])
    ax.set_ylim([-2 * radSc, 2 * radSc])
    return "One part of grouped radial plot complete"


def compute_sobol_for_different_metrics(
    usace_strategy_states, npv_maximization_states, sample_strategy_states, pars
):
    problem = {
        "num_vars": 12,
        "names": [
            "b",
            "epsilon",
            "phi_conc",
            "A",
            "alpha",
            "beta",
            "d",
            "eta",
            "l",
            "W",
            "xi",
            "delta",
        ],
        "bounds": [
            [0, 0.113 * 10**-3],
            [0, 0.1],
            [0.7 * 193.8357, 1.3 * 193.8357],
            [0.7 * 669000, 1.3 * 669000],
            [1.0, 1.05],
            [0.95, 1.0],
            [-0.05, 0.05],
            [0.7 * 824, 1.3 * 824],
            [0.176, 0.264],
            [0.7 * 500000, 1.3 * 500000],
            [0, 0.1],
            [0.01, 0.07],
        ],
    }
    # Generate Sobol Samples
    param_samples = saltelli.sample(problem, 2**12)
    # extract each parameter for input into the beach nourishment problem
    b_samples = param_samples[:, 0]
    epsilon_samples = param_samples[:, 1]
    phi_conc_samples = param_samples[:, 2]
    A_samples = param_samples[:, 3]
    alpha_samples = param_samples[:, 4]
    beta_samples = param_samples[:, 5]
    d_samples = param_samples[:, 6]
    eta_samples = param_samples[:, 7]
    l_samples = param_samples[:, 8]
    W_samples = param_samples[:, 9]
    xi_samples = param_samples[:, 10]
    delta_samples = param_samples[:, 11]

    groups = {
        "Geo-physical": ["b", "epsilon", "phi_conc"],
        "Socio-economic": ["A", "alpha", "beta", "d", "eta", "l", "W", "xi", "delta"],
    }
    (
        npv_usace,
        benefits_usace,
        costs_usace,
        investment_costs_usace,
        damage_costs_usace,
        reliability_usace,
        bcr_each_sow_usace,
        bcr_pass_fail_each_sow_usace,
    ) = calculate_beach_nourishment_metrics_strategy(
        usace_strategy_states, param_samples, pars
    )

    (
        npv_cutler,
        benefits_cutler,
        costs_cutler,
        investment_costs_cutler,
        damage_costs_cutler,
        reliability_cutler,
        bcr_each_sow_cutler,
        bcr_pass_fail_each_sow_cutler,
    ) = calculate_beach_nourishment_metrics_strategy(
        npv_maximization_states, param_samples, pars
    )

    (
        npv_sample,
        benefits_sample,
        costs_sample,
        investment_costs_sample,
        damage_costs_sample,
        reliability_sample,
        bcr_each_sow_sample,
        bcr_pass_fail_each_sow_sample,
    ) = calculate_beach_nourishment_metrics_strategy(
        sample_strategy_states, param_samples, pars
    )

    SA_npv_usace = sobol.analyze(problem, npv_usace, print_to_console=True)
    SA_reliability_usace = sobol.analyze(
        problem, reliability_usace, print_to_console=True
    )
    SA_costs_usace = sobol.analyze(problem, costs_usace, print_to_console=True)
    SA_benefits_usace = sobol.analyze(problem, benefits_usace, print_to_console=True)
    SA_npv_cutler = sobol.analyze(problem, npv_cutler, print_to_console=True)
    SA_reliability_cutler = sobol.analyze(
        problem, reliability_cutler, print_to_console=True
    )
    SA_costs_cutler = sobol.analyze(problem, costs_cutler, print_to_console=True)
    SA_benefits_cutler = sobol.analyze(problem, benefits_cutler, print_to_console=True)
    SA_costs_sample = sobol.analyze(problem, costs_sample, print_to_console=True)
    SA_benefits_sample = sobol.analyze(problem, benefits_sample, print_to_console=True)
    SA_npv_sample = sobol.analyze(problem, npv_sample, print_to_console=True)
    SA_reliability_sample = sobol.analyze(
        problem, reliability_sample, print_to_console=True
    )
    return (
        SA_npv_usace,
        SA_reliability_usace,
        SA_costs_usace,
        SA_benefits_usace,
        SA_npv_cutler,
        SA_reliability_cutler,
        SA_costs_cutler,
        SA_benefits_cutler,
        SA_costs_sample,
        SA_benefits_sample,
        SA_npv_sample,
        SA_reliability_sample,
    )


def plot_radial_chord_plot_npv_reliability(
    SA_npv_sample, SA_reliability_sample, SA_npv_usace, SA_reliability_usace
):
    parameters_list = [
        "b",
        "epsilon",
        "phi_conc",
        "A",
        "alpha",
        "beta",
        "d",
        "eta",
        "l",
        "W",
        "xi",
        "delta",
    ]
    parameters_proxy = [
        "SLR \n coefficient",
        "Increase in \n storm induced \n erosion",
        "SLR \n damage \n coefficient",
        "Baseline \n property \n valuation",
        "% change in \n property value \n w. beach width",
        "% change in \n property value \n w. SLR",
        "% increase in \n baseline \n property \n value",
        "Land \n value",
        "Property \n tax \n rate",
        "Non-physcial \n value at risk",
        "% increase\nin cost\nof sand",
        "Discount \nrate",
    ]
    groups = {
        "Geo-physical": ["b", "epsilon", "phi_conc"],
        "Socio-economic": ["A", "alpha", "beta", "d", "eta", "l", "W", "xi", "delta"],
    }
    # Create the main figure and subplots
    fig, ax = plt.subplots(
        2, 2, gridspec_kw={"hspace": 0.01, "wspace": 0.01}, figsize=(20, 20)
    )
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.1, wspace=0.1)

    # Titles for rows and columns
    row_titles = ["Satisficing Strategy", "US Army Corps Strategy"]
    column_titles = ["Net Present Value (NPV)", "Reliability"]

    # Assuming grouped_radial is defined elsewhere
    grouped_radial(
        SA_npv_sample,
        parameters_list,
        parameters_proxy,
        ax[0, 0],
        groups=groups,
        threshold=0.0025,
        varNameMult=1.525,
        gpNameMult=2.0,
    )
    grouped_radial(
        SA_reliability_sample,
        parameters_list,
        parameters_proxy,
        ax[0, 1],
        groups=groups,
        threshold=0.0025,
        varNameMult=1.535,
        gpNameMult=2.0,
    )
    grouped_radial(
        SA_npv_usace,
        parameters_list,
        parameters_proxy,
        ax[1, 0],
        groups=groups,
        threshold=0.0025,
        varNameMult=1.525,
        gpNameMult=2.0,
    )
    grouped_radial(
        SA_reliability_usace,
        parameters_list,
        parameters_proxy,
        ax[1, 1],
        groups=groups,
        threshold=0.0025,
        varNameMult=1.525,
        gpNameMult=2.0,
    )

    # Adding titles above each column
    for j in range(2):
        ax[0, j].set_title(column_titles[j], fontsize=36, fontweight="bold")

    # Adding labels to subplots
    ax[0, 0].text(
        0.06,
        0.97,
        "A)",
        transform=ax[0, 0].transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[0, 1].text(
        0.06,
        0.97,
        "B)",
        transform=ax[0, 1].transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[1, 0].text(
        0.06,
        0.97,
        "C)",
        transform=ax[1, 0].transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[1, 1].text(
        0.06,
        0.97,
        "D)",
        transform=ax[1, 1].transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Adding titles to the left of each row
    for i in range(2):
        fig.text(
            0.001,
            0.75 - i * 0.5,
            row_titles[i],
            fontsize=36,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )

    # Set spines (borders) around each subplot
    for i in range(2):
        for j in range(2):
            for spine in ax[i, j].spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(2)

    # Adjust overall layout
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.03)

    # Save the main figure
    plt.savefig("sobol_main_plot.jpeg", dpi=100, bbox_inches="tight")
    plt.close()

    # Now create a custom legend using PIL
    # Define the size of the legend
    legend_width = 1200
    legend_height = 200

    # Create a blank white image for the legend
    legend_img = Image.new("RGB", (legend_width, legend_height), color="white")
    draw = ImageDraw.Draw(legend_img)

    # Try to load a font
    try:
        # Try to get a nice font - adjust path as needed for your system
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/Helvetica.ttc"  # macOS alternative
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\Arial.ttf"  # Windows alternative

        title_font = ImageFont.truetype(font_path, 30)
        percent_font = ImageFont.truetype(font_path, 28)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        percent_font = ImageFont.load_default()

    # Draw "Sensitivity Indices" text
    draw.text(
        (100, 100), "Sensitivity\nIndices", fill="black", font=title_font, anchor="mm"
    )

    # Define positions
    first_order_center_x = 400
    total_order_center_x = 700
    second_order_center_x = 1000
    center_y = 100

    # First Order
    # Small circle
    draw.ellipse(
        (
            first_order_center_x - 100 - 7,
            center_y - 7,
            first_order_center_x - 100 + 7,
            center_y + 7,
        ),
        fill="gray",
        outline="gray",
    )
    # Large circle
    draw.ellipse(
        (
            first_order_center_x + 30 - 45,
            center_y - 45,
            first_order_center_x + 30 + 45,
            center_y + 45,
        ),
        fill="gray",
        outline="gray",
    )
    # Text
    draw.text(
        (first_order_center_x - 100, center_y - 60),
        "0.25%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (first_order_center_x + 30, center_y - 60),
        "94%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (first_order_center_x - 35, center_y + 60),
        "First Order",
        fill="black",
        font=title_font,
        anchor="mm",
    )

    # Total Order
    # Small circle
    draw.ellipse(
        (
            total_order_center_x - 100 - 7,
            center_y - 7,
            total_order_center_x - 100 + 7,
            center_y + 7,
        ),
        outline="black",
        width=2,
    )
    # Large circle
    draw.ellipse(
        (
            total_order_center_x + 30 - 45,
            center_y - 45,
            total_order_center_x + 30 + 45,
            center_y + 45,
        ),
        outline="black",
        width=2,
    )
    # Text
    draw.text(
        (total_order_center_x - 100, center_y - 60),
        "0.25%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (total_order_center_x + 30, center_y - 60),
        "97%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (total_order_center_x - 35, center_y + 60),
        "Total Order",
        fill="black",
        font=title_font,
        anchor="mm",
    )

    # Second Order
    # Line
    draw.line(
        (
            second_order_center_x - 100 - 20,
            center_y,
            second_order_center_x - 100 + 20,
            center_y,
        ),
        fill="gray",
        width=6,
    )
    # Rectangle
    draw.rectangle(
        (
            second_order_center_x + 30 - 40,
            center_y - 25,
            second_order_center_x + 30 + 40,
            center_y + 25,
        ),
        fill="gray",
    )
    # Text
    draw.text(
        (second_order_center_x - 100, center_y - 60),
        "0.25%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (second_order_center_x + 30, center_y - 60),
        "43%",
        fill="black",
        font=percent_font,
        anchor="mm",
    )
    draw.text(
        (second_order_center_x - 35, center_y + 60),
        "Second Order",
        fill="black",
        font=title_font,
        anchor="mm",
    )

    # Save the legend
    legend_img.save("sobol_legend.jpeg", quality=95)

    # Now combine the two images using PIL
    # Open the two images
    main_img = Image.open("sobol_main_plot.jpeg")
    legend_img = Image.open("sobol_legend.jpeg")

    # Resize the legend to match the width of the main image
    legend_width = main_img.width
    legend_height = int(main_img.width * (legend_img.height / legend_img.width))
    legend_img_resized = legend_img.resize((legend_width, legend_height), Image.LANCZOS)

    # Create a new image with enough height for both
    new_height = main_img.height + legend_height
    combined_img = Image.new("RGB", (main_img.width, new_height), color="white")

    # Paste the main image at the top
    combined_img.paste(main_img, (0, 0))

    # Paste the legend below the main image
    combined_img.paste(legend_img_resized, (0, main_img.height))
    # Save the original directory
    original_dir = os.getcwd()
    # Go one folder up
    parent_dir = os.path.dirname(original_dir)
    os.chdir(parent_dir)
    figures_dir = os.path.join(parent_dir, "figures")
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # Go into "Figures" folder
    os.chdir(figures_dir)
    # Create "Main Figures" folder if it doesn't exist
    main_figures_dir = os.path.join(figures_dir, "main_figures")
    if not os.path.exists(main_figures_dir):
        os.mkdir(main_figures_dir)
    os.chdir(main_figures_dir)
    print("PWD", os.getcwd())
    # Save the combined image
    combined_img.save("sobol_test_10_new.jpeg", quality=95)
    os.chdir(original_dir)


def is_significant(value, confidence_interval, threshold="conf"):
    if threshold == "conf":
        return value - abs(confidence_interval) > 0
    else:
        return value - abs(float(threshold)) > 0


def plot_scarf(ax, data, title, panel_label, colors):
    # Add panel label (A, B, C, D)
    ax.text(
        -0.07,
        0.5,
        panel_label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="center",
        ha="center",
    )

    # Reshape data for visualization
    data_array = np.array(data)

    # Create a colormap
    cmap = {0: colors[0], 1: colors[1], 2: colors[2]}

    # Plot each element as a colored segment
    for i, val in enumerate(data_array):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=cmap[int(val)]))

    # Set plot properties
    ax.set_xlim(0, len(data_array))
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=22)  # Increased title font size
    ax.set_yticks([])
    ax.set_xticks([])


def plot_strategies(
    usace_strategy,
    npv_max_strategy,
    threshold_satisficing_strategy,
    robustness_satisficing_strategy,
):
    colors = ["#80b1d3", "#8dd3c7", "#fb8072"]  # Blue, Teal, Salmon
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), gridspec_kw={"hspace": 0.4})
    # Plot each strategy with panel labels
    plot_scarf(axs[0], usace_strategy, "USACE Strategy", "A)", colors)
    plot_scarf(axs[1], npv_max_strategy, "NPV Maximizing Strategy", "B)", colors)
    plot_scarf(
        axs[2],
        threshold_satisficing_strategy,
        "Threshold Satisficing Strategy with maximum NPV",
        "C)",
        colors,
    )
    plot_scarf(
        axs[3],
        robustness_satisficing_strategy,
        "Robustness Satisficing Strategy",
        "D)",
        colors,
    )
    # Add a legend at the bottom of the plot with simplified labels
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=4, label="Do Nothing"),
        Line2D([0], [0], color=colors[1], lw=4, label="Nourish"),
        Line2D([0], [0], color=colors[2], lw=4, label="Retreat"),
    ]

    # Place the legend below the subplots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.02),
    )

    # Add more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Adjust the rect to make room for legend
    original_dir = os.getcwd()
    # Go one folder up
    parent_dir = os.path.dirname(original_dir)
    os.chdir(parent_dir)
    figures_dir = os.path.join(parent_dir, "figures")
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    # Go into "Figures" folder
    os.chdir(figures_dir)
    # Create "Main Figures" folder if it doesn't exist
    main_figures_dir = os.path.join(figures_dir, "supplementary_figures")
    if not os.path.exists(main_figures_dir):
        os.mkdir(main_figures_dir)
    os.chdir(main_figures_dir)
    plt.savefig("Strategies.png", dpi=600)
    os.chdir(original_dir)
