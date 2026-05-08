# Guide des taches restantes (etudiants)

Le code est complet. Il reste les taches qui demandent un travail humain
ou un acces a la webcam. Cette liste est exhaustive : ne rien rajouter
au code, sauf si l'enseignante le demande.

---

## Tache 1 -- Construire le dataset (priorite haute)

**Objectif final :** corpus mixte couvrant la contrainte de l'enonce
(>= 10 personnes x >= 20 images), constitue de :

- **2 identites etudiants** capturees a la webcam (vous-memes),
- **>= 8 identites publiques** integrees automatiquement depuis le dataset
  Essex Faces94.

### 1.a -- Integrer le dataset public (a faire en premier)

Le script `scripts/download_dataset.py` telecharge l'archive Essex Faces94
(~19 MB), passe chaque image dans le pipeline d'alignement Haar (`src.alignment.align_face`)
et sauvegarde les 12 premiers sujets dans `captures/<id>/img_NNN.jpg`
(jusqu'a 25 images par personne). Le script est idempotent : un sujet
deja integre n'est pas recalcule, vos captures webcam ne seront jamais
ecrasees.

```bash
python scripts/download_dataset.py
```

Sortie attendue : 12 dossiers `captures/<id_numerique>/` contenant chacun
~20 images 128x128 grises.

**Dataset utilise :** [Essex Faces94](https://cswww.essex.ac.uk/mv/allfaces/faces94.html)
(Dr Libor Spacek, University of Essex Vision Group, 1996). 153 sujets,
20 images frontales chacun, fond uni vert, variation d'expression
naturelle. **Licence :** libre pour la recherche academique non
commerciale (cite la source dans le rapport). Le script utilise un
miroir GitHub stable (BYU "Foundations of Applied Mathematics") car le
serveur Essex est intermittent ; en dernier recours il bascule sur
l'URL Essex officielle.

**A citer dans le rapport / les slides :** "Une partie du corpus
provient de la base Essex Faces94 (Vision Group, Univ. of Essex)
distribuee pour la recherche academique." Avec lien :
https://cswww.essex.ac.uk/mv/allfaces/faces94.html

### 1.b -- Capturer vos 2 identites a la webcam

Pour les **2 etudiants** (et eux uniquement, le reste du corpus est
deja integre via 1.a) :

1. Activer l'environnement et lancer la capture :
   ```bash
   python scripts/capture.py Prenom_Nom
   ```
2. Suivre le protocole exige par l'enonce, a appliquer **integralement
   sur vos propres captures** car les images Faces94 ne couvrent que la
   variation d'expression :
   - **5 images** face neutre (regard droit, expression detendue)
   - **5 images** avec expressions (sourire, surprise, yeux plisses)
   - **5 images** rotation laterale (+/- 15°, 2-3 a gauche, 2-3 a droite)
   - **5 images** eclairage modifie (lampe a gauche, a droite, contre-jour attenue)
3. La fenetre n'enregistre que si le bandeau passe en vert (`OK`).
   Si elle reste rouge ("Visage non detecte"), eclairer le visage / s'eloigner / oter les lunettes.

### 1.c -- Verification

```bash
python scripts/build_dataset.py
```

La sortie doit afficher >= 14 personnes (12 publiques + 2 webcam) et
>= 280 vecteurs. Si une personne a moins d'images exploitables,
recommencer la capture (1.b) pour celle-ci.

### Questions courantes

- **"Visage non detecte" en permanence** -> Haar frontal a besoin d'un visage de
  face avec un bon eclairage. Tester sans lunettes, cheveux degages.
- **Webcam non trouvee** -> verifier qu'aucune autre application ne l'utilise
  (Teams, Zoom, navigateur).
- **Image floue / sombre** -> stabiliser la camera, ajouter une lampe.

---

## Tache 2 -- Calibrer le seuil

Le seuil par defaut (`src/config.py`, `DEFAULT_THRESHOLD = 0.18`) est arbitraire.
Sur le seul corpus Faces94 (12 sujets, 258 vecteurs), le balayage
donne un optimum vers **0.225 - 0.30** (accuracy ~0.89, rejet 0-3%).
Apres ajout de vos 2 captures webcam, refaire le sweep : la valeur
optimale peut changer parce que vos images presentent une variation de
pose et d'eclairage absente de Faces94.

```bash
python scripts/evaluate.py --sweep
```

Identifier la valeur de `seuil` qui maximise l'accuracy en leave-one-out tout
en gardant un taux de rejet raisonnable (~5-15% sur des visages connus).

Mettre cette valeur dans `src/config.py`. Re-evaluer :

```bash
python scripts/evaluate.py
```

Conserver la matrice de confusion et les metriques pour le rapport.

---

## Tache 3 -- Analyser les erreurs

Pour chaque erreur d'identification (case hors-diagonale dans la matrice),
repondre dans le rapport aux 4 questions de l'enonce :

1. Quelle personne a ete confondue avec quelle autre ?
2. Le vecteur Snake (16 derniers coefficients) est-il different entre les deux personnes ?
3. Les distances geometriques (14 premiers coefficients) sont-elles similaires ?
4. Quel parametre modifier pour corriger cette confusion ?
   (ex. augmenter `kappa` du snake, changer le seuil, ajouter une distance...)

Astuce: pour comparer deux personnes A et B, charger `dataset.csv`
dans Excel / pandas, calculer la moyenne des vecteurs de A et de B,
et lister les coordonnees ou la difference est la plus marquee.

---

## Tache 4 -- Rapport (a redacter par les etudiants)

Plan suggere :

1. **Introduction** : contexte, objectifs, contraintes (pas d'apprentissage).
2. **Pipeline** : schema bloc, role de chaque module.
3. **Detection et alignement** : Haar, choix de l'angle, equalisation.
4. **Contour actif** : formulation de Kass-Witkin, role de alpha/beta/kappa,
   resultats sur quelques visages.
5. **Vecteur caracteristique** : choix des 14 distances, justification du
   nombre de rayons (16), normalisation.
6. **Identification** : recherche NN, calcul de confiance, top-3.
7. **Resultats** : matrice de confusion, courbe accuracy = f(seuil),
   metriques par classe, exemples reussis et echecs.
8. **Analyse des erreurs** : 2-3 cas detaille, parametres a ajuster.
9. **Limites et perspectives** : sensibilite a la pose, a l'eclairage,
   pistes d'amelioration sans deep learning (ex. LBP, descripteurs SIFT).
10. **Conclusion**.

---

## Tache 5 -- Presentation (slides)

Plan court (10-12 slides) :

1. Titre + equipe + module.
2. Probleme et contraintes.
3. Pipeline en une figure.
4. Detection + alignement (avant / apres).
5. Snake : equation, evolution sur un visage (4 etapes).
6. Vecteur 30D : schema des 14 distances + 16 rayons.
7. Identification : exemple top-3 reussi.
8. Identification : exemple ambigu / erreur.
9. Matrice de confusion + courbe seuil.
10. Demo (ou capture d'ecran) de la porte simulee.
11. Limites + ouvertures.
12. Conclusion / Q&R.

---

## Tache 6 -- Repetition demo

Avant la soutenance, verifier en conditions reelles :

```bash
python scripts/main.py
```

- Avec une personne enregistree -> porte verte, nom affiche.
- Avec une personne inconnue -> porte rouge, "Inconnu" affiche.
- Verifier que `Q` ferme proprement les deux fenetres.

---

## Ce qui a ete fait pour vous

- Detection + alignement du visage (`src/alignment.py`)
- Snake from scratch (Kass-Witkin, `src/snake.py`)
- Detection de points caracteristiques (`src/landmarks.py`)
- Construction du vecteur 30D (`src/features.py`)
- Persistance CSV (`src/dataset.py`)
- Identification + score de confiance + top-K (`src/identify.py`)
- Leave-one-out + matrice de confusion + sweep de seuil (`src/evaluation.py`)
- Simulation porte Tkinter (`src/door_sim.py`)
- Application temps reel webcam + porte (`scripts/main.py`)
- Scripts de capture, encodage, evaluation
