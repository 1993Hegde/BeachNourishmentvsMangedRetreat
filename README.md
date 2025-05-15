# Hegde-etal_2025

## Considering retreat can lead to more robust adaptation strategies in nourishment-dependent coastal communities


Prabhat Hegde <sup> 1 *</sup>, Adam Pollack <sup>1</sup>, Vikrant Vaze <sup>1</sup>, Klaus Keller <sup>1</sup>

<sup>1</sup> Thayer School of Engineering, Dartmouth College; Hanover, 03755, USA.

<sup>*</sup> corresponding author : Prabhat.Hegde.TH@dartmouth.edu

## Abstract

Many coastal towns rely on cyclical beach nourishment to mitigate erosion risks. However, rising flood risks threaten the long-term viability of this nourishment strategy. This raises the question: When to  retreat? Decision making agencies and research on decision frameworks largely overlook this question, neglecting key considerations such as uncertainties that affect optimal timing and navigating trade-offs between multiple objectives.  We address this previously overlooked question by designing and implementing a framework that identifies Pareto-optimal strategies balancing multiple objectives under uncertainty, demonstrated through a case study. Our results show that neglecting retreat as a planning lever makes it impossible to balance multiple objectives and meet the regulatory benefit-cost threshold. Furthermore, accounting for uncertainty necessitates retreat at least two decades earlier than deterministic approaches to meet the benefit-cost threshold. Our framework can help planners devise more robust long-term adaptation strategies by informing the choice of if and when to retreat. 

## Overview of this repository
This repository includes the code and instructions for reproducing the main analysis in the paper Considering retreat can be crucial to design robust adaptation strategies in nourishment-dependent coastal communities.

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
   
    f) Run all the three python files in this folder in sequence with `python id_benchmark_decisions.py && python mordm_analyses.py && python visualize_results.py`. Alternatively, you cah choose to run them one after another.


Please contact [Prabhat Hegde](mailto:Prabhat.Hegde.TH@dartmouth.edu) if you have any problems following these instructions. 


