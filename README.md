# Quantifying Landmine Classifier Stability via BCa Confidence Intervals

## Overview
This project investigates the minimum sample size ($n$) required to achieve stable predictive performance for landmine classification using a Random Forest classifier. It utilizes the **Bias Corrected and Accelerated ($BC_a$)** bootstrap methodology to evaluate estimator stability across both non-parametric and parametric designs.

The study is based on the research paper included in this repository (`CIS_project.pdf`), which concludes that $n=200$ is the optimal balance between classification certainty and operational safety for this specific domain.

## Project Structure
* **`project.py`**: The main execution script. It handles data loading, model training, synthetic consistency checks, BCa bootstrap experiments (parametric and non-parametric), and results plotting.
* **`data/`**: Contains the dataset (`Land mines.csv`).
* **`images/`**: Output directory where the script saves the generated plots (e.g., performance shrinkage, bias comparison).
* **`CIS_project.pdf`**: The full academic report detailing the methodology and results.

## Data
The dataset used is the **Land Mines** dataset (Kahraman et al.), which contains data from passive magnetic sensors.
* **Source:** UCI Machine Learning Repository.
* **Features:** Voltage (V), Height (H), Soil Type (S).
* **Target:** Mine Type (5 distinct classes).

## Installation
The project was developed using **Python 3.11.14**. The following libraries are required:

`pip install numpy pandas scikit-learn scipy seaborn matplotlib`

Key Library Versions 
* scikit-learn: 1.7.2
* scipy: 1.16.2
* numpy: 2.3.5
* pandas: 2.2.3

To reproduce the experiments and generate the plots, run the main script:
`python project.py`
or the notebook
`project.ipynb`

What the script does:
1. Consistency Check: Runs a simulation on synthetic data ($N=10,000$) to verify that the $BC_a$ intervals achieve the nominal coverage probability (95%).
2. Data Processing: Loads Land mines.csv, performs one-hot encoding on the 'Soil Type' feature, and prepares the data for the Random Forest model.
3. Hyperparameter Tuning: Performs a Grid Search (10-fold CV) to find optimal Random Forest parameters.
4. Bootstrap Experiments: * Non-Parametric: Resamples $(X, y)$ pairs to test stability against sampling variability.
5. Parametric: Fixes $X$ and generates new $y$ labels based on model probabilities to test stability against label noise.Output: Saves results to CSVs and generates comparison plots in the `images/` directory.

## Methodology
The stability of the classifier is quantified using the $BC_a$ Confidence Interval, which is second-order accurate ($O(n^{-1})$). This method is chosen over standard percentile intervals because it explicitly corrects for:
* Bias ($w$): The difference between the bootstrap median and the original estimate.
* Acceleration ($a$): The rate of change of the standard error with respect to the parameter.

## Results
Experimental results indicate that while stability improves with increased data, marginal gains diminish significantly beyond $n=200$.
* At $n=200$: The non-parametric 95% CI reaches a practical width of 0.0914 with a mean ROC AUC of 0.8220.
* Parametric vs. Non-Parametric: The study found the non-parametric approach to be more robust. The parametric approach suffered from numerical collapse at higher sample sizes due to generator bias.
For full details, refer to `CIS_project.pdf`.

## Author
**Andreas Larsson** December 2025
