# tp-sem-1
Contains scripts for a project based on the 2014 SemDis task (lexical substitution)

# consigne
Analyse sémantique automatique du langage naturel M2 Linguistique  
Informatique – Univ. Paris Diderot, Marie Candito  
marie.candito@linguist.univ-paris-diderot.fr  

**Evaluation de vecteurs de mots par une tâche de substitution lexicale**

**A rendre par mail pour le 22 octobre 2017**:

- un zip, contenant votre programme marinecourtin_luigiliu.py2
- un rapport marinecourtin_luigiliu.pdf explicitant 
	+ la tâche à effectuer, 
	+ la méthode utilisée pour y parvenir, 
	+ ainsi que les résultats obtenus.

**1 La tâche de substitution lexicale SemDis 2014**

Etudiez la définition de la tâche https://www.irit.fr/semdis2014/fr/task1.html, ainsi que le jeu de données de test fourni, les réponses gold, le format attendu des réponses, les métriques d’évaluation et le script d’évaluation.

Les hyperparamètres à tester sont donc :

- L’espace vectoriel à utiliser : soit FREDIST soit FRWAK-SKIPGRAM (NB: fredist se présente comme un thésaurus précalculé à partir de vecteurs distributionnels creux. On considèrera que les XXX plus proches voisins de chaque mot sont les seules dimensions non nulles de ces vecteurs)

- CIBLE_INCLUSE : un booléen pour inclure ou pas au contexte le mot cible lui-même

- La taille F de la fenêtre de contexte (on considèrera F mots pleins à gauche + F mots pleins à droite)
	+ prévoir une valeur « infinie » : tous les mots pleins de la phrase


**2 Expérimentations avec vecteurs de mots**

Implémentez la méthode ci-dessous, utilisez le module d’évaluation fourni par les utilisateurs, et testez différentes configurations d’hyperparamètres.

**2.1 Génération des candidats substituts :**

On propose d’utiliser comme candidats substitut pour un mot cible ses n plus proches voisins dans un espace vectoriel de mots, de même catégorie morpho-syntaxique que le mot cible. On testera deux ressources :

**2.2 Score pour les candidats substituts :**

- FREDIST : (Henestroza Anguiano & Denis, 2011) : les plus proches voisins sont déjà calculés, téléchargeable ici :
https://gforge.inria.fr/projects/fredist/ .

- FRWAK-SKIPGRAM : Des vecteurs obtenus avec word2vec, par J.P. Fauconnier, et disponibles ici : http://fauconnier.github.io/\#data, plus précisément nous utiliserons ceux calculés avec skip-gram negative
sampling, sur le corpus FRWAK, en version lemmatisés, avec catégories morphosyntaxiques, 700 dim, cutoff de 50 :
http://embeddings.org/frWac\_postag\_no\_phrase\_700\_skip\_cut50.bin (utilisez le package de manipulation de .bin word2vec, cf. http://fauconnier.github.io/\#data)

Vous utiliserez la similarité entre le substitut et un Continuous Bag of Words, i.e. le vecteur obtenu en sommant les vecteurs des mots de contexte. Plus précisément, pour un couple (c=mot cible à substituer, s=candidat substitut) :

 Soit CXT l’ens. des mots pleins dans le contexte de c (limitable à 2F mots pleins, F étant un hyperparamètre), soit Z le vecteur obtenu en faisant une somme des vecteurs des mots de CXT, on utilise simplement score(vecteur(s), Z).

Une variante est d’inclure ou pas dans le CXT le mot cible lui-même, cf. le candidat substitut peut être un voisin plus ou moins proche de c.
