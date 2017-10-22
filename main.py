#! /usr/bin/env python
# coding:utf-8

# Prérequis :
# pip install Cython word2vec

# Auteurs
#
# Marine Courtin, Université Sorbonne Nouvelle
# Luigi Liu, Université Paris Nanterre
#
# marine.courtin@etud.sorbonne-nouvelle.fr
# luigi.plurital@gmail.com

import argparse, word2vec, sys, numpy, codecs, time
import semdis_eval
from lexsub import *

# VARIABLES GLOBALES
n_candidats = 10
F_max = 6
OVER_SAMPLING = 1

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	parser.add_argument('resfile', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
	parser.add_argument('goldfile', type=str, help='fichier de réponse gold')
	parser.add_argument('-o','--outfile', type=str, help='fichier de réponse en sortie',default = 'tmp')
	parser.add_argument('-v','--verbose', help='voir des détails en temps réel', action="store_true")
	parser.add_argument('-r','--restype', type=int, help="le type de ressource lexciale employée : 0 pour FRWAC (par défault), 1 pour FRDIC, 2 pour HYBRIDE, 3 pour FRWAC avec une nouvelle méthode", default = 0) #ajout d'un 3eme mode d'utilisation des ressources
	args = parser.parse_args()

	if args.verbose :
		print (args)

	# chargement des ressources lexicales
	model = word2vec.load(args.resfile)

	with codecs.open(args.infile, encoding = 'utf-8') as f :
		for CIBLE_INCLUSE in [False, True] :
			for F in range(0, F_max) :
				t1 = time.time()
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

						# a. transformer tous les mots dans CTX en vecteurs dont la liste retournée est E
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
						elif args.restype == 3 :

							# nouvelle proposition basée sur les ressources word2vec
							candidats, scores = gen_subs_new (model, CTX, c, c_pos, n = 10)

						else : # i.e. args.restype == 0

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
					print u"La dernière bouclee s'est terminée en", get_duration(t1_secs = t1, t2_secs = time.time())
					if args.restype == 1 : exit()
