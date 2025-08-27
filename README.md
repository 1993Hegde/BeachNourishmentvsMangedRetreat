# Hegde-etal_2025

## Timing managed retreat for robust coastal adaptation strategies


Prabhat Hegde <sup> 1 *</sup>, Adam Pollack <sup>1</sup>, Vikrant Vaze <sup>1</sup>, Klaus Keller <sup>1</sup>

<sup>1</sup> Thayer School of Engineering, Dartmouth College; Hanover, 03755, USA.

<sup>*</sup> corresponding author : Prabhat.Hegde.TH@dartmouth.edu

## Abstract

Many coastal towns rely on cyclical beach nourishment to mitigate erosion risks. However, rising flood risks threaten the long-term viability of this nourishment strategy. This raises the question: When to  retreat? Decision making agencies and research on decision frameworks largely overlook this question, neglecting key considerations such as uncertainties that affect optimal timing and navigating trade-offs between multiple objectives.  We address this previously overlooked question by designing and implementing a framework that identifies Pareto-optimal strategies balancing multiple objectives under uncertainty, demonstrated through a case study. Our results show that neglecting retreat as a planning lever makes it impossible to balance multiple objectives and meet the regulatory benefit-cost threshold. Furthermore, accounting for uncertainty necessitates retreat at least two decades earlier than deterministic approaches to meet the benefit-cost threshold. Our framework can help planners devise more robust long-term adaptation strategies by informing the choice of if and when to retreat. 

## Overview of this repository
This repository includes the code and instructions for reproducing the main analysis in the paper <u>*Considering retreat can be crucial to design robust adaptation strategies in nourishment-dependent coastal communities.*</u>

For a coastal town, the key uncertainties considered, observed state variables, metrics of innterest and their dependencies are mapped in the systems diagram below. 

<img src="https://github.com/1993Hegde/BeachNourishmentvsMangedRetreat/blob/91ccd005cb76555a4fd6546c68d3164f870eda00/F1_SystemRelationships.jpeg" alt="image_alt" width="700" />

For comparison, the system relationships are consistent with <u><em>Cutler, Emma M., Mary R. Albert, and Kathleen D. White. "Tradeoffs between beach nourishment and managed retreat: Insights from dynamic programming for climate adaptation decisions." Environmental Modelling & Software 125 (2020): 104603.</em></em></u><sup><a href="https://www.sciencedirect.com/science/article/pii/S1364815219303639">[1]</a></sup><sup><a href="https://github.com/emcutler/coastal-management">[2]</a></sup>

A detailed shoreline protection plan proposed by the U.S. Army Corps of Engineers for the location used in the case study, please refer to <u><em>District, USACE Jacksonville. "St. Lucie County, Florida. Coastal Storm Risk Management Project. Final Integrated Feasibility Study and Environmental Assessment." US Army Corps of Engineers, Jacksonville District, Jacksonville, Florida (2017).</em></u><sup><a href="https://www.saj.usace.army.mil/Missions/Civil-Works/Shore-Protection/St-Lucie-County/">[3]</a></sup>

## Reproduce our experiment
You can follow the instructions below to reproduce all results reported in the manuscript and supplementary materials. For this experiment, reproduction does not imply bit-wise reproducibility. You should obtain similar quantitative results and figures, with all metrics within a tolerance of 5%. 

To reproduce:
1. Clone this repository into a local directory.
2. Get your environment ready. We recommend using `mamba`.
   
    a) Change directory to env with `cd env`.
   
    b) Run mamba `env create -f env.yml` or replace `mamba` with `conda`.
   
    c) Once this is complete, run `mamba activate bn_test`.
   
    d) Change to the main directory using `cd ..`.
   
    e) Change to the workflow directory with `cd workflow`.
   
    f) Run all the three python files in this folder in sequence with `python id_benchmark_decisions.py && python mordm_analyses.py && python visualize_results.py`. Alternatively, you can choose to run them one after another.

The high-leve functionality of each script is mentioned below:
| Script Name | Description |
| --- | --- |
| `python id_benchmark_decisions.py` | Run discrete dynamic program for best guess parameter values, identify NPX Maximizing strategy and USACE approximation| 
| `mordm_analyses.py` | Runs Multi Objective Optimization for 2 and 5 objectives, with and without uncertainty |
| `visualize_results.py` | Generates all the figures reported in the paper |

Please contact [Prabhat.Hegde.TH@dartmouth.edu](mailto:Prabhat.Hegde.TH@dartmouth.edu) if you have any problems following these instructions. 


