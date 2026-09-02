# Benchmarks de performance MPQP

Cette suite mesure les performances des opérations importantes de MPQP sans
effectuer d'appel réseau ni exécuter de job chez un fournisseur quantique.

Les benchmarks actuels couvrent :

- la construction d'un circuit ;
- le calcul de sa profondeur ;
- sa conversion en matrice ;
- l'import et l'export QASM 2 ;
- le regroupement de chaînes de Pauli.

Tous les résultats générés localement sont placés dans `.benchmarks/`. Ce dossier
est ignoré par Git afin d'éviter de créer des fichiers à la racine du dépôt ou de
les ajouter accidentellement à un commit.

## Installation

Depuis la racine du dépôt MPQP :

```bash
python -m pip install -r requirements-dev.txt
```

## Lancer les benchmarks localement

Exécution simple avec affichage des résultats dans le terminal :

```bash
python -m pytest benchmarks --benchmark-only
```

Exécution avec export JSON :

```bash
python -m pytest benchmarks --benchmark-only --benchmark-json=.benchmarks/results/benchmark-results.json
```

Enregistrer une mesure de référence :

```bash
python -m pytest benchmarks --benchmark-only --benchmark-save=baseline
```

`pytest-benchmark` enregistre cette référence dans un sous-dossier de
`.benchmarks/` correspondant au système et à la version de Python.

Comparer une nouvelle exécution à cette référence et échouer si la médiane est
plus lente de 15 % :

```bash
python -m pytest benchmarks --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=median:15%
```

Générer les graphiques SVG à partir des mesures sauvegardées :

```bash
pytest-benchmark compare --histogram=.benchmarks/reports/benchmark-histogram
```

La commande normale `python -m pytest` continue uniquement à exécuter les tests
fonctionnels présents dans `tests/`.

## Workflow GitHub Actions

Le workflow `.github/workflows/benchmarks.yml` peut être lancé :

- automatiquement après un push ou un merge sur `main` ;
- manuellement depuis **Actions → Benchmarks dashboard → Run workflow**.

Pour une exécution manuelle, les paramètres suivants sont disponibles :

| Paramètre | Description |
| --- | --- |
| `runner` | Machine utilisée pour mesurer les performances. |
| `python_version` | Version de Python utilisée. |
| `commit_ref` | Branche, tag ou SHA à mesurer. Vide signifie le commit courant. |
| `save` | Indique si le résultat doit être ajouté à l'historique permanent. |
| `gh_pages_branch` | Branche cible dans `MPQP-PrivateBenchmark`. |

Les fichiers JSON temporaires du workflow sont créés dans :

```text
.benchmarks/results/benchmark-results.json
```

Ils sont également disponibles pendant 90 jours dans les artefacts de
l'exécution GitHub Actions.

## Fonctionnement de `save`

Avec `save=false`, le workflow :

1. exécute les benchmarks ;
2. produit le fichier JSON sous `.benchmarks/results/` ;
3. compare les résultats avec l'historique existant ;
4. ajoute un résumé dans GitHub Actions ;
5. ne pousse aucune modification dans `MPQP-PrivateBenchmark`.

Avec `save=true`, il effectue les mêmes opérations puis ajoute la mesure et met à
jour la page web dans :

```text
ColibrITD-SAS/MPQP-PrivateBenchmark
```

Les exécutions automatiques sur `main` sont toujours sauvegardées, même si le
paramètre `save` n'est pas présent sur ce type d'événement.

## Organisation du dépôt MPQP-PrivateBenchmark

Les résultats sont séparés par système d'exploitation et version de Python afin
de ne comparer que des mesures prises dans un environnement équivalent :

```text
MPQP-PrivateBenchmark/
└── dev/
    └── bench/
        ├── Linux-CPython-3.12/
        │   ├── index.html
        │   └── data.js
        └── Windows-CPython-3.12/
            ├── index.html
            └── data.js
```

La page Linux/Python 3.12 est disponible à l'adresse suivante lorsque GitHub
Pages est activé :

```text
https://colibritd-sas.github.io/MPQP-PrivateBenchmark/dev/bench/Linux-CPython-3.12/
```

Chaque page contient l'évolution temporelle de chaque benchmark ainsi que les
informations du commit correspondant. L'historique est limité aux 100 dernières
mesures par graphique.

