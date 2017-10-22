#! /usr/bin/env python
# coding:utf-8

# Prérequis :
# pip install Cython word2vec

# auteurs
# Marine Courtin (Université Paris Sorbonne Nouvelle)
# Luigi Liu (Université Paris Nanterre)

# todos :
# 1. + expension sémantique
# 2. rapport en Markdown :
# 	Vous ferez un petit rapport réexpliquant la tâche, la méthode, et commentant vos résultats.
# 	Votre programme doit contenir une aide en ligne (l’option –h doit indiquer comment utiliser le programme).

# références :
# [1] Melamud, O., Levy, O., Dagan, I., & Ramat-Gan, I. (2015, June). A Simple Word Embedding Model for Lexical Substitution. In VS@ HLT-NAACL (pp. 1-7).
# [2] Desalle, Y., Navarro, E., Chudy, Y., Magistry, P., & Gaume, B. (2014). BACANAL: Balades Aléatoires Courtes pour ANAlyses Lexicales Application à la substitution lexicale. In TALN-20 2014: Atelier SEMDIS.
#

import argparse, word2vec, sys, numpy, codecs, time
import semdis_eval
from lexsub import *

# VARIABLES GLOBALES
n_candidats = 10
F_max = 10
OVER_SAMPLING = 2

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	parser.add_argument('resfile', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
	parser.add_argument('goldfile', type=str, help='fichier de réponse gold')
	parser.add_argument('-o','--outfile', type=str, help='fichier de réponse en sortie',default = 'tmp')
	parser.add_argument('-v','--verbose', help='voir des détails en temps réel', action="store_true")
	parser.add_argument('-r','--restype', type=int, help="le type de ressource lexciale employée : 0 pour FRWAC (par défault), 1 pour FRDIC, 2 pour HYBRIDE", default = 0) #ajout d'un 3eme mode d'utilisation des ressources
	args = parser.parse_args()

	if args.verbose :
		print (args)

	# chargement des ressources lexicales
	model = word2vec.load(args.resfile)

	with codecs.open(args.infile, encoding = 'utf-8') as f :
		for CIBLE_INCLUSE in [False, True] :
			for F in range(0, F_max) :
				t = time.time()
				if not F and not CIBLE_INCLUSE : continue
				with codecs.open(args.outfile, 'w', encoding = 'utf-8') as fout :
					for line in f :

						# lecture des colonnes de fichier
						id, c, c_pos, c_position, sentence = line.split(u'\t')
						tokens = [t.split(u'/') for t in sentence.split()]

						# préparation et nettoyage du contexte plein
                                                c_position_new, tokens_full = rm_stopword_from_tokens(tokens, cat_full, c_position)
                                                overwindowing,CTX = windowing (tokens_full, c_position_new, F, CIBLE_INCLUSE)
                                                CTX = clean_ctx(CTX)
                                                Z = continous_bag_words(model, CTX)

						# géneration des substituts potentiels
						if args.restype == 1:
							candidats = generateSubstitutes(c, c_pos, n_candidats)
						elif args.restype == 2:
							# on prend les 100 premiers résultats dans FREDIST comme substituts potentiels
							# nb de substituts choisi arbitraire
							candidats = generateSubstitutes(c, c_pos)
							try: #équivalent à la condition 'if candidats == None :'
								candidats = [candidat + u"_" + c_pos for candidat in candidats]
							except TypeError: #cible pas dans FREDIST, on retombe sur r = 0
								candidats, scores = \
								generateSubstitutes_w2v(model, c, c_pos, OVER_SAMPLING * n_candidats)

							candidats, scores = sort_response(model, candidats, Z)
							candidats = candidats[0 : n_candidats]
						else :
							candidats, scores = \
							generateSubstitutes_w2v(model, c, c_pos, OVER_SAMPLING * n_candidats)

							# ordonnancement de la liste des substituts proposés par le contexte
							candidats, scores = sort_response(model, candidats, Z)
							candidats = candidats[0 : n_candidats]

						# sorties
						if args.verbose :
							show_infobox (id, c, c_pos, c_position, sentence, F, CIBLE_INCLUSE, CTX, args.restype == 1)
							print('candidats de substituant proposés : ')
							for i, cand in enumerate(candidats) : print(u'\t{} : {}'.format(i, cand))
						if candidats :
							export_substituants (id, c, c_pos, candidats, fout)

					f.seek(0)
					# évaluation
					print (u'(F, CIBLE_INCLUSE) = ({}, {})'.format(F, CIBLE_INCLUSE))
					s = semdis_eval.SemdisEvaluation(args.goldfile)
					s.evaluate(args.outfile, metric = 'all', normalize = True)
					# pour le moment la solution basée sur FRDIC n'emploie pas le contexte
					# pas intéressant de boucler avec (F, CIBLE_INCLUSE) différents
					print "... done in", get_duration(t1_secs = t, t2_secs = time.time())
					if args.restype == 1 : exit()
