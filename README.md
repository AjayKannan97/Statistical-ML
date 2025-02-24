# Statistical Machine Learning

## Overview
This repository contains implementations of various **Statistical Machine Learning (ML) algorithms** using **Python**. The focus is on understanding fundamental statistical concepts and applying them to machine learning problems.

## Project Objectives
- Implement **supervised and unsupervised** statistical ML models.
- Apply **probabilistic reasoning** to real-world datasets.
- Explore **Bayesian methods**, **regression techniques**, and **classification models**.
- Compare model performance using statistical evaluation metrics.

## Technologies Used
- **Programming Language**: Python
- **Libraries Used**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `statsmodels`

## Repository Structure
```
Statistical-ML/
│── datasets/               # Sample datasets used for training/testing
│── notebooks/              # Jupyter Notebooks for experiments & visualizations
│── src/                    # Source code for implementing ML models
│── results/                # Performance analysis and model evaluation
│── bayesian_models.py       # Implementation of Bayesian learning models
│── regression_models.py     # Linear and logistic regression models
│── classification_models.py # Decision Trees, SVM, and Naive Bayes
│── clustering_models.py     # K-Means, Hierarchical, and DBSCAN clustering
│── README.md               # Project documentation
```

## Implemented Models
- **Regression Models**:
  - Linear Regression
  - Logistic Regression
  - Ridge & Lasso Regression

- **Classification Models**:
  - Decision Trees
  - Support Vector Machines (SVM)
  - Naïve Bayes Classifier

- **Bayesian Learning**:
  - Bayesian Networks
  - Gaussian Naïve Bayes
  - Markov Chain Monte Carlo (MCMC)

- **Clustering Algorithms**:
  - K-Means
  - Hierarchical Clustering
  - DBSCAN

## Setup Instructions
### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### Running the Models
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AjayKannan97/Statistical-ML.git
   cd Statistical-ML
   ```
2. **Run Jupyter Notebook for interactive exploration**:
   ```bash
   jupyter notebook
   ```
3. **Run specific models**:
   ```bash
   python regression_models.py
   python classification_models.py
   python clustering_models.py
   ```

## Results & Observations
- **Regression models** accurately predicted continuous variables with minimal errors.
- **Classification models** showed high accuracy on structured datasets.
- **Bayesian learning models** effectively handled uncertainty and probabilistic inference.
- **Clustering techniques** successfully grouped unlabeled data points.

## Applications
- Predictive analytics in **finance, healthcare, and marketing**.
- Text classification and **spam detection**.
- Customer segmentation using **unsupervised learning**.
- Anomaly detection in **cybersecurity**.

## Contributors
- **Ajay Kannan**  
- [Add collaborators if applicable]  

## License
This project is for educational purposes. Please give appropriate credit if used.

---
For any questions, contact **Ajay Kannan** at ajaykannan@gmail.com.  
