import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pickle
import colorcet as cc
from mordm_functions import *
import pandas as pd
from matplotlib.ticker import FuncFormatter


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
    print(os.getcwd())
    plt.savefig("time_possibilities_new_convergence.png", dpi=600, bbox_inches="tight")
    return fig


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
