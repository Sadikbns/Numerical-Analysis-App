# Outil d'Analyse Numérique

## Description du Projet

Application graphique développée en **Python** avec **Tkinter** permettant d'appliquer les principaux algorithmes d'analyse numérique étudiés en cours et TP.

L'application est structurée en **trois axes** principaux :

- **Axe 1** : Résolution des équations non linéaires
- **Axe 2** : Résolution des systèmes linéaires
- **Axe 3** : Interpolation et Approximation

---

## Fonctionnalités

### Axe 1 — Équations Non Linéaires
- Méthodes : Dichotomie (Bissection), Point Fixe, Newton
- Calcul de dérivées, vérification de continuité
- Test de stabilité et de contraction
- Visualisation graphique de la fonction et de la convergence
- Tableau des itérations détaillé

### Axe 2 — Systèmes Linéaires
- Algorithmes directs : Gauss (pivot partiel/total), LU, Cholesky
- Algorithmes itératifs : Jacobi, Gauss-Seidel, Relaxation
- Saisie de matrices jusqu'à 4×4
- Calcul de déterminant, normes induites
- Visualisation de la convergence (pour les méthodes itératives)

### Axe 3 — Interpolation et Approximation
- Interpolation polynomiale : Lagrange, Newton
- Approximation : Moindres Carrés, Chebyshev
- Descente de Gradient
- Calcul des normes discrètes et continues (L1, L2, L∞)
- Analyse d'erreur (discrète et continue)
- Graphiques interactifs + tableaux

---

## Structure du Projet
Numerical Analysis App/
├── main_screen.py
├── axe1_screen.py
├── axe2_screen.py
├── axe3_screen.py
├── modules/
│   ├── chap 1/
│   ├── chap 2 and 3/
│   └── chap 4 and 5/
└── README.md


---

## Technologies Utilisées

- **Python 3**
- **Tkinter** + **ttk** (interface graphique)
- **NumPy** (calculs numériques)
- **SymPy** (manipulation symbolique)
- **Matplotlib** (graphiques intégrés)
- **Scipy** (interpolation supplémentaire)

---

## Comment exécuter l'application ?

1. Placez tous les fichiers dans le même dossier
2. Assurez-vous que les modules sont dans `modules/chap X/`
3. Exécutez la commande suivante :

```bash
python main_screen.py
```
---

## Membres du Groupe 
- BAATOUT Mohamed Amine 232331396111
- BEN SADI Elsadik 232331351708
- LISRI Akram 232331200815
- LOUNI Mohammed Said 232331499716

---
## Remarques

L'interface est entièrement interactive
Les résultats sont accompagnés de visualisations claires (matplotlib)
L'application est responsive