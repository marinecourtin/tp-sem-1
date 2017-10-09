#! /usr/bin/env python
# coding:utf-8

# Prérequis :
# pip install Cython word2vec

# auteurs
# Marine Courtin (Université Paris Sorbonne Nouvelle)
# Luigi Liu (Université Paris Nanterre, MoDyCo)

# todos :
# 2. élaborer un test des hyperparamètres F, CIBLE_INCLUSES pour avoir un premier bilan -> utiliser la méthode grid search ?
#     L -> c'est-à-dire ? itérer sur une double boucle avec (F,CIBLE_INCLUSES) dans toutes les valeurs pertinentes ?
#     M -> il me semble qu'il y a une fonction qui existe pour ça : from sklearn.grid_search import ParameterGrid +
#     on a aussi l'air de pouvoir trouver automatiquement la combinaison de cesparamètres qui maximise le score,
#     je ne sais pas si tu as déjà utilisé ça : http://scikit-learn.sourceforge.net/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
#     L -> ça peut être utile : deux limites, tout de même.
#          1. dépendence de la bibliothéque sklearn
#          2. on veux aussi observer la courbe score(F,CIBLE_INCLUSE) pour mieux comprendre la tâche
#          le code a été refactorisé pour que tu puisses appliquer plus facilement gridsearch -> donc, allez y !
#
# 3. pondérer les vecteurs par les poids obtenus par la TF-IDF sur un corpus de français
# 4. introduire 2-ème solution sur FREDIST : (Henestroza Anguiano & Denis, 2011) : les plus proches voisins sont déjà
#                                  calculés, téléchargeable ici : https://gforge.inria.fr/projects/fredist/
# 7. rapport écrit suivant la consigne :
# 	Vous ferez un petit rapport réexpliquant la tâche, la méthode, et commentant vos résultats.
# 	Votre programme doit contenir une aide en ligne (l’option –h doit indiquer comment utiliser le
# 	programme).
# 8. Toute idée d’amélioration est la bienvenue
# 9. (implémentation facultative) repas au Crous

# références :
# [1] Melamud, O., Levy, O., Dagan, I., & Ramat-Gan, I. (2015, June). A Simple Word Embedding Model for Lexical Substitution. In VS@ HLT-NAACL (pp. 1-7).
# [2] Desalle, Y., Navarro, E., Chudy, Y., Magistry, P., & Gaume, B. (2014). BACANAL: Balades Aléatoires Courtes pour ANAlyses Lexicales Application à la substitution lexicale. In TALN-20 2014: Atelier SEMDIS.
#

# proposition
# la somme des vecteurs de mot dans le contexte plein semble faire perdre
# plus d'informations qu'elle permet d'agréer. nous proposons de ne pas sommer
# les vecteurs, mais conisdérer les vecteurs du contexte plein comme un ensemble
# de coordonnées qui permettent de localiser le bon mot cible
#
# note : si nous voulons traiter les vecteurs de mots dans le contexte plein comme
# des coordonnées -> Algorithme de Gram-Schmidt est utile pour trouver
# les coordonnées de bonne formation (Algèbre linéaire)
#
# soit les vecteurs du mots pleins dans le contexte v_x1, v_x2, ..., v_xn,
# vx_i : i-ème vecteur du contexte plein, i in [1,n], x pour noter conte'x'te
# vc : vecteur du mot cible
# nous définissions un pi comme le produit scalaire entre v_xi et vc
# pi := produit_scalair(v_xi, vc)
# P = [p1,p2,...,pN] est une liste d'inicateur de longueur N
# qi := produit_scalaire(v_xi, vy)
# Q = [q1,q2,...,qN]
# vy est un vecteur de mot dans le vocabualaire
#
# la nouvelle mesure de sililarité comme
# A. produit_scalaire(P,Q) version non-normalisée
# B. produit_scalaire(P,Q) version normalisée
#
# reférence : idée est proche du "match filter" dans le domaine du traitement du signal

import argparse, word2vec, sys, numpy, codecs
import semdis_eval
from lexsub import *

# VARIABLES GLOBALES
n_candidats = 10
F_max = 10
OVER_SMAPLING = 2

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	parser.add_argument('resfile', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
	parser.add_argument('goldfile', type=str, help='fichier de réponse gold')
	parser.add_argument('-o','--outfile', type=str, help='fichier de réponse en sortie',default = 'tmp')
	parser.add_argument('-v','--verbose', help="increase output verbosity", action="store_true")
	args = parser.parse_args()

	# chargement des ressources lexicales
	model = word2vec.load(args.resfile)

	with codecs.open(args.infile, encoding = 'utf-8') as f :
		for CIBLE_INCLUSE in [False, True] :
			for F in range(0, F_max) :
				if not F and not CIBLE_INCLUSE : continue
				with codecs.open(args.outfile, 'w', encoding = 'utf-8') as fout :
					for line in f :
						# lecture des cololones de fchier
						id, c, c_pos, c_position, sentence = line.split(u'\t')
						tokens = [t.split(u'/') for t in sentence.split()]

						# géneration des substituatnts
						candidats, scores = generateSubstitutes_w2v(model, c, c_pos, OVER_SMAPLING * n_candidats)

						# préparation et nettoyage du contexte pleine
						c_position_new, tokens_full = rm_stopword_from_tokens(tokens, cat_full, c_position)
						CTX = windowing (tokens_full, c_position_new, F, CIBLE_INCLUSE)
						CTX = clean_ctx(CTX)
						Z = continous_bag_words(model, CTX)
						# ordonnancement de la liste des substituants proposés par le contexte
						candidats, scores = sort_response(model, candidats, Z)
						candidats         =                      candidats[0:n_candidats]

						# sorties
						if args.verbose : show_infobox (id, c, c_pos, c_position, sentence, F, CIBLE_INCLUSE, CTX)
						if candidats : export_substituants (id, c, c_pos, candidats, fout)

					f.seek(0)
					# évaluation
					print (u'(F, CIBLE_INCLUSE) = ({}, {})'.format(F, CIBLE_INCLUSE))
					s = semdis_eval.SemdisEvaluation(args.goldfile)
					s.evaluate(args.outfile, metric = 'all', normalize = True)
