# Neuro-Layered Adaptive Memory (NLAM)

**Neuro-Layered Adaptive Memory (NLAM)** est un modèle conçu pour résoudre le problème du **Catastrophic Forgetting** en Continual Learning. Il s'inspire du fonctionnement de la mémoire humaine et intègre une structure hiérarchique pour gérer efficacement la rétention et l'adaptation des connaissances.

---

## **Description du projet**
L'apprentissage séquentiel des modèles neuronaux pose un défi majeur : **l'oubli catastrophique**. Lorsqu'un modèle apprend de nouvelles tâches, il tend à écraser les connaissances précédemment acquises, ce qui empêche une généralisation efficace sur le long terme.

Le modèle **NLAM (Neuro-Layered Adaptive Memory)** propose une approche innovante en structurant la mémoire en trois niveaux :
- **Mémoire à court terme (STM) :** Capture les informations récentes et temporaires.
- **Mémoire intermédiaire (IM) :** Extrait des représentations globales et contextuelles.
- **Mémoire à long terme (LTM) :** Stocke des représentations latentes stables pour éviter la perte d'informations critiques.

Le modèle est en phase de développement et de test.

---

## **Pourquoi NLAM ?**
- **Évite l'oubli catastrophique** en structurant la mémoire de manière hiérarchique.
- **Facilite le transfert de connaissances** en stockant uniquement les informations essentielles dans la mémoire à long terme.
- **Modèle dynamique et adaptatif** qui sélectionne et ajuste ses représentations en fonction des nouvelles tâches.
- **Inspiré des neurosciences cognitives**, simulant la manière dont l’humain apprend et consolide les connaissances.

---

## **Architecture du modèle**
NLAM repose sur 5 composants clés :

1. **Système de mémoire hiérarchique (STM, IM, LTM)**
2. **Mécanisme de contrôle (Gating Mechanism)** qui active dynamiquement les couches de mémoire.
3. **Mécanisme d'attention de la mémoire (Memory Attention Mechanism)** qui ajuste les contributions des différentes mémoires.
4. **Mécanisme de transfert de connaissances (Knowledge Transfer Mechanism)** qui consolide les informations stables vers la LTM.
5. **Mécanisme d'oubli adaptatif (Forgetting Mechanism)** qui empêche l'accumulation d'informations inutiles.

Chaque composant joue un rôle essentiel pour permettre au modèle d’apprendre **sans effacer** les connaissances acquises.
