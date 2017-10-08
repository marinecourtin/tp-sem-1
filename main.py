# coding:utf-8

# Prérequis :
# pip install Cython word2vec

# auteurs
# Marine Courtin (Paris Sorbonne Nouvelle)
# Luigi Liu (Paris Nanterre, MoDyCo)

# todos :
# 1. intégrer le script d'évaluation https://www.irit.fr/semdis2014/data/semdis2014_evaluation.tar.gz
# 2. éloborer un test des hyperparamètres F, CIBLE_INCLUSES pour avoir un premier bilan
# 3. pondérer les vecteurs par les poids obtenus par la TF-IDF sur un corpus de français
# 4. introduire 2-ème solution sur FREDIST : (Henestroza Anguiano & Denis, 2011) : les plus proches voisins sont déjà
#                                  calculés, téléchargeable ici : https://gforge.inria.fr/projects/fredist/
# 5. correction des lemmes non prises en charge à cause de leur ambiguïté, voir fixme
# 6. rapport écrit suivant la consigne :
# 	Vous ferez un petit rapport réexpliquant la tâche, la méthode, et commentant vos résultats.
# 	Votre programme doit contenir une aide en ligne (l’option –h doit indiquer comment utiliser le
# 	programme).
# 7. Toute idée d’amélioration est la bienvenue
# 8. (implémentation facultative) repas au Crous

import argparse, word2vec, sys, numpy, codecs

# VARIABLES GLOBALES
cat_full = ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV'] # POS pour les mots pleins (de Marine)

# FONCTIONS
def conv_pos(pos_melt) :

	"""
	'ADV'  -> 'adv'
	'NC'   -> 'n'
	'VINF' -> 'v'
	"""

	if pos_melt != 'ADV' : pos_melt = pos_melt[0]
	return pos_melt.lower()

def repr_sentence(sentence, c_position) :

	"""
	args
	sentence : str, la phrase d'entrée
	c_position : str or int, la position du mot cible dans la phrase, c_position > 0
	"""

	ret = u''
	for cursor, t in enumerate(sentence.split()) :
		if cursor + 1 == int(c_position) : ret += u'#'
		ret += t.split(u'/')[0] + u' ' # t = [token, pos, lemme]
	return ret

def generate_response(w2v_model, vec, pos_desired, n = 10):

	"""
	génération des substituants basé sur la similarité de cosinus,
	cette fonction est adaptée de
	model.cosine() et model.generate_response() provenant du script
	https://github.com/danielfrg/word2vec/blob/master/word2vec/wordvectors.py
	"""

	# cosinus(v1, v2) = pruduit_scalaire(v1, v2) / norm(v1) / norm(v2)
	#                 = produit_scalaire(v1 / norm(v1), v2 / norm(v2))
	#
	#                   si norm(v1), norm(v2) > 0

	# normalisation de vecteur v2, i.e. 'vec'
	if numpy.linalg.norm(vec) != 0 : vec = vec / numpy.linalg.norm(vec)

	# les vecteurs dans la ressource de J-P Fauconnier est prénormalisé
	# ils sont rangés dans une matrice, i.e. vectors
	# vectors = [v10
	#	     v11
	#            v12
	#             :
	#             :
	#            v1N]
	#
	# v1i : vector de dimension 700 x 1 qui représente le i-ème mot du vocabulaire V
	# N : taille du vocabulaire V
	# ]
	#
	# dot(vectors, v2) = [produit_scalaire(v10,v2)
	#                     produit_scalaire(v11,v2)
	#                     produit_scalaire(v12,v2)
	#	               :
	#		       :
	#                     produit_scalaire(v1N,v2)]
	#
	#                  = [cosinus(v10,v2)
	#                     cosinus(v11,v2)
	#                     cosinus(v12,v2)
	#			:
	#			:
	#                     cosinus(v1N,v2)]
	#
	#                   = la smilarité cosinus entre le mot représenté par le vecteur v2 et chacun des mots dans le vocabulaire
	metrics = numpy.dot(w2v_model.vectors, vec)
	indexes_best = numpy.argsort(metrics)[::-1][1:]

	# sélectionner les n meilleures candidats selon la métrique de cosinus
	# les candidats doivent la catégorie POS spécifié par 'pos_desired'
	cnt = 0
	candidats = []
	scores = []
	for cursor, i in enumerate(indexes_best) :
		word_pos = w2v_model.vocab[i]
		# ex: word_pos = 'intéresser_v' -> pos = 'v'
		pos = word_pos.split('_')[-1]
		if pos == pos_desired :
			candidats.append(w2v_model.vocab[i])
			scores.append(metrics[i])
			cnt += 1
			if cnt == n : break # on ne prend que les n meilleurs

	return candidats, scores

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	parser.add_argument('resfile', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
	args = parser.parse_args()

	# chargement des ressources lexicales
	model = word2vec.load(args.resfile)

	# hyperparamètres (à varier par la suite)
	CIBLE_INCLUSE = True
	F = 3

	with codecs.open(args.infile, encoding = 'utf-8') as f :
		for line in f :
			id, c, c_pos, c_position, sentence = line.split(u'\t')
			tokens = [t.split(u'/') for t in sentence.split()]
			# filtrage des mots vides par catégorie morpho-syntaxique
			tokens_full = []
			j = 0
			for i, t in enumerate(tokens) :
				token, pos, lemme = t
				if i + 1 == int(c_position) :
					c_position_new = j
				if pos in cat_full :
					tokens_full.append(t)
					j += 1
			# construction du contexte par méthode de fenêtrage
			CTX = []
			for i, t in enumerate(tokens_full) :
				if F < 0 :
					CTX.append(t)
				elif i == c_position_new :
					if CIBLE_INCLUSE :
						CTX.append(t)
				elif abs(i - c_position_new) <= F :
					CTX.append(t)
			# vectorisation des mots en contexte
			Z = None
			for ctx in CTX :
				token, pos, lemme = ctx
				lemme_pos = lemme.lower() + u'_' + conv_pos(pos)
				try :
					if not Z :
						Z = model[lemme_pos]
					else :
						Z += model[lemme_pos]
				except :
					# fixme : des ambiguïtés dans la forme lemmatisée
					# comme vivre|voir dans vit/V/vivre|voir à résoudre
					# afin de mieux profiter des ressources lexicales
					continue

			# génération des substituants et leurs scores de similarités avec le contexte
			candidats, scores = generate_response(model, Z, c_pos)

			# affichage
			print (u'instance id : {}'.format(id))
			print (u'target token : {}'.format(c))
			print (u'target POS : {}'.format(c_pos))
			print (u'full sentence : \n\t{}'.format(repr_sentence(sentence, c_position)))
			print (u'\nCTX(F = {}, CIBLE_INCLUSE = {}) : '.format(F, CIBLE_INCLUSE))
			print (u'{:>32s} {:>6s} {:>32s}'.format(u'Token',u'POS',u'Lemme'))
			for ctx in CTX :
				print (u'{:>32s} {:>6s} {:>32s}'.format(*ctx))
			print (u'\n{:>20s} {:>17s}'.format(u'SUBSTITUANTS',u'SCORES'))
			for c, s in zip(candidats, scores) : print (u'{:>20s} {:>16.15f}'.format(c, s))
			print(u'\n')

