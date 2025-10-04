# Task 8 — K-Means Clustering

This project demonstrates **unsupervised machine learning** using the **K-Means clustering algorithm**.  
It works with the `Mall_Customers.csv` dataset (from the task PDF) or can generate synthetic demo data if no dataset is provided.

## 📂 Project Structure
kmeans-task8/
│── kmeans_clustering.py    # Main script
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── data/                   # Place your CSV datasets here
│── outputs/                # Generated plots & results
│── .venv/                  # Virtual environment (not pushed to GitHub)

## ⚙️ Installation & Setup

1. Clone the repository
   git clone https://github.com/BSyoga/kmeans-task8.git
   cd kmeans-task8

2. Create and activate virtual environment

   Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

## ▶️ Usage

Run with your dataset  
Place `Mall_Customers.csv` inside the `data/` folder, then run:
   python kmeans_clustering.py --data data/Mall_Customers.csv

Run with demo dataset (if no CSV is provided)
   python kmeans_clustering.py

Specify number of clusters manually
   python kmeans_clustering.py --data data/Mall_Customers.csv --k 4

## 📊 Outputs

All results are saved in the **outputs/** folder:
- elbow.png → Elbow method plot (Inertia vs K)
- silhouette.png → Silhouette scores vs K
- clusters.png → Final 2D cluster visualization (via PCA)
- clustered_data.csv → Dataset with assigned cluster labels

## 🧰 Dependencies
- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib

Install them using:
   pip install -r requirements.txt

## ✨ Notes
- If your dataset has columns like **Annual Income** and **Spending Score**, the script will automatically detect numeric columns.
- If the script can’t find numeric columns, it falls back to demo data.
- Use the **silhouette score** or **elbow method** to determine the best number of clusters.

**Author:** YOGA

