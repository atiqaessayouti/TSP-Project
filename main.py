import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# S'assurer que les dossiers locaux sont reconnus
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tsp_model import TSPInstance
from solvers.hill_climbing import hill_climbing_best
from solvers.recuit_simule import simulated_annealing

# 1. Préparation de l'instance (50 villes)
np.random.seed(42)
coords = np.random.rand(50, 2) * 100
instance = TSPInstance(coords)

# 2. Partie Statistique (30 Runs) - Pour le tableau du rapport
def run_experiments(n_runs=30):
    hc_results = []
    sa_results = []
    
    print(f"⏳ Lancement des {n_runs} tests pour les statistiques...")
    for i in range(n_runs):
        _, d_hc, _ = hill_climbing_best(instance)
        _, d_sa, _ = simulated_annealing(instance)
        hc_results.append(d_hc)
        sa_results.append(d_sa)
    
    print("\n--- RÉSULTATS STATISTIQUES ---")
    print(f"HC | Meilleur: {min(hc_results):.2f} | Moyenne: {np.mean(hc_results):.2f} | Std: {np.std(hc_results):.2f}")
    print(f"SA | Meilleur: {min(sa_results):.2f} | Moyenne: {np.mean(sa_results):.2f} | Std: {np.std(sa_results):.2f}\n")

# 3. Partie Visualisation (1 Run détaillé)
def show_visuals():
    print("📈 Génération des graphiques de convergence et de la Map...")
    
    # Exécution simple pour capturer l'historique
    hc_tour, hc_dist, hc_hist = hill_climbing_best(instance)
    sa_tour, sa_dist, sa_hist = simulated_annealing(instance, T0=200, alpha=0.995, max_iter=8000)

    # --- Graphique de Convergence ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hc_hist, label="Hill Climbing", color='red')
    plt.plot(sa_hist, label="Recuit Simulé", color='blue')
    plt.title("Convergence (Baisse de la distance)")
    plt.xlabel("Itérations")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)

    # --- Graphique de la Map (Chemin) ---
    plt.subplot(1, 2, 2)
    # On ajoute la première ville à la fin pour fermer la boucle
    sa_tour_plot = list(sa_tour) + [sa_tour[0]]
    coords_plot = coords[sa_tour_plot]
    plt.plot(coords_plot[:, 0], coords_plot[:, 1], 'bo-', mfc='red', markersize=5)
    plt.title(f"Meilleur trajet (SA): {sa_dist:.2f}")
    
    plt.tight_layout()
    plt.show() # Ouvre la fenêtre avec les deux graphes

if __name__ == "__main__":
    run_experiments(n_runs=30)
    show_visuals()