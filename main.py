# coding:utf-8

# Prérequis :
# pip install Cython word2vec

# auteurs
# Marine Courtin (Paris Sorbonne Nouvelle)
# Luigi Liu (Paris Nanterre, MoDyCo)

# todos :
# 2. élaborer un test des hyperparamètres F, CIBLE_INCLUSES pour avoir un premier bilan -> utiliser la méthode grid search ?
#     -> c'est-à-dire ? itérer sur une double boucle avec (F,CIBLE_INCLUSES) dans toutes les valeurs pertinentes ?
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
# je nous mets deux articles ci-dessous présentant leur solution pour cette tâche, ça permet de comprendre
# comment les chercheurs voient la chose 
# [1] Melamud, O., Levy, O., Dagan, I., & Ramat-Gan, I. (2015, June). A Simple Word Embedding Model for Lexical Substitution. In VS@ HLT-NAACL (pp. 1-7).
# [2] Desalle, Y., Navarro, E., Chudy, Y., Magistry, P., & Gaume, B. (2014). BACANAL: Balades Aléatoires Courtes pour ANAlyses Lexicales Application à la substitution lexicale. In TALN-20 2014: Atelier SEMDIS.
# 
# pour le premier article, on voit que les mots dans le contexte ne sont jamais sommées pour faire qqch.
# ils sont impliqués dans une métrique de manière 'non-linéaire'

# demande de précisions :
# sur la consigne, on ne voit pas très bien elle veut soit (pour le cas de FRWAC)
# 1. on génére des canadidats à partir de l'information du vecteur du  mot cible et en ordonner la liste par 
#    leur similarité avec son contexte plein 
# soit
# 2. trouver directement les mots les plus proches du contexte plein dans le vocaubulaire
# 
# je ne sais pas si tu vois ce que je voulais dire. en fait, j'ai implémenté 2., ça fait que mes candidats sont
# préalablement proche du contexte. après avoir lu des articles, je pense que c'est 1 qui est mieux. je 
# le corrigerai. Mais la consigne -> elle te paraît claire sur ce point-là ?


# attention : the script d'évaluation n'est compatible qu'avec python 2
#             en conséquence, notre projet code est contraint d'être développé
#             pour python 2.

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

# VARIABLES GLOBALES
dico_lemme_pos_fix = {'compris_a':'comprendre_v'}
cat_full = ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV'] # POS pour les mots pleins (de Marine)

# FONCTIONS
def rm_pos(lemme_pos) :
	if u'_' not in lemme_pos : return lemme_pos
	tmp = lemme_pos.split(u'_')
	return u'_'.join(tmp[0 : len(tmp) - 1])

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

def sort_response (w2v_model, candidats, Z) :

	scores = []
	scores_candidats = []
	if numpy.linalg.norm(Z) != 0 : vec = Z / numpy.linalg.norm(Z)
	for candidat in candidats :
		score = numpy.dot(w2v_model[candidat], Z)
		scores_candidats.append([score, candidat])
	scores_candidats = sorted(scores_candidats, key=lambda x : x[0], reverse = True)
	if scores_candidats :
		scores    = [x[0] for x in scores_candidats]
		candidats = [x[1] for x in scores_candidats]
	return candidats, scores

def generate_response(w2v_model, vec, pos_desired, n = 10) :

	if vec is None or not w2v_model : return None,None

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


def generateSubstitutes(c, c_pos, n=15):
	"""
	sélectionne les candidats substituts à la cible. La fonction est basée sur
	l'utilisation de la ressource FREDIST disponible à l'adresse :
	https://gforge.inria.fr/projects/fredist/
	args : cible, pos de la cible, nombre de candidats
	output : liste des n candidats les plus similaires

	"""
	c_pos = c_pos.upper()
	# print(c, c_pos)
	with codecs.open("./thesauri-1.0/thesaurus_french_"+c_pos+'.txt', encoding = 'utf-8') as f :
		for line in f:
			line = line.split("\t")
			term, subs = (line[0].split("|")[1], line[1:][:n])
			if term == c:
				candidats = [sub.split("|")[1] for sub in subs]
				candidats = [can.split(":")[0] for can in candidats]
				print(term, c, candidats)
		if not candidats:
			return None
	return candidats

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	parser.add_argument('resfile', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
	parser.add_argument('outfile', type=str, help='fichier de réponses en sortie')
	args = parser.parse_args()

	# chargement des ressources lexicales
	model = word2vec.load(args.resfile)

	# hyperparamètres (à varier par la suite)
	CIBLE_INCLUSE = True
	F = 3
	n_candidats = 10

	fout = codecs.open(args.outfile, 'w', encoding = 'utf-8')

	with codecs.open(args.infile, encoding = 'utf-8') as f :
		for line in f :
			id, c, c_pos, c_position, sentence = line.split(u'\t')
			tokens = [t.split(u'/') for t in sentence.split()]

			# I. génération des substituants à partir du mot cible 'c' :
			c_lemme_pos = c + '_' + c_pos
			# réparation manuelle des lemmes non-renseignés ou mal rensignés
			if c_lemme_pos in dico_lemme_pos_fix.keys() :
				c_lemme_pos = dico_lemme_pos_fix[c_lemme_pos]
			candidats, scores = generate_response(model, model[c_lemme_pos], c_pos, n_candidats)

			# II. prétratiment du contexte
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

			# nettoyage du contexte plein
			CTX2 = []
			for ctx in CTX :
				token, pos, lemmes = ctx

				# suppression de '*' dans les lemmes
				lemmes = lemmes.replace(u'*',u'')

				# traitier l'ambiguiïté dans les lemmes
				if u'|' in lemmes :
					for lemme in lemmes.split(u'|') :
						if lemme :
							CTX2.append([token,pos,lemme])
				else :
					CTX2.append([token, pos, lemmes])
			CTX = CTX2

			# vectorisation des mots en contexte
			# t0 :   Z = v_0
			# t1 :   Z = v_0 + v_1
			# t2 :   Z = v_0 + v_1 + v_2
			# tn-2 : Z = v_0 + v_1 + ... + v_n-2
			# tn-1 : Z = v_0 + v_1 + ... + v_n-2 + v_n-1 -> la somme souhaitée
			# n : le nombre de mots pleins dans le contexte
			# v_i : i-ème vecteur de mot visité pour le calcul de leur somme
			# + : addition vectorielle
			# print(CTX) ; exit()
			Z = None
			for ctx in CTX :
				token, pos, lemme = ctx
				lemme_pos = lemme.lower() + u'_' + conv_pos(pos)
				try :
					if not Z :
						# initier "la somme vectorielle" Z par la valeur du premier vecteur visité
						# comme ça Z est typé comme un vecteur
						Z = model[lemme_pos]
					else :
						# une fois initiation faite, on accumule des vecteurs par
						# l'addition vectorielle ou pointwise addition
						#
						# note: lorsque le symbole + et utilisé comme un opérateur binaire
						# dans, par exemple a + b, '+' est devenue une addition vectorielle
						# si a, b sont du type vectoriel et de la même taille (longueur)
						Z += model[lemme_pos]
				except :
					continue

			# II. ordonnancement de la liste des candidats
			#     par leur similarité avec le contexte plein représeté en
			#     Continous Bag of Words


			# génération des substituants et leurs scores de similarités avec le contexte
			candidats, scores = sort_response(model, candidats, Z)

			# affichage pour les humains
			print (u'instance id : {}'.format(id))
			print (u'target token : {}'.format(c))
			print (u'target POS : {}'.format(c_pos))
			print (u'full sentence : \n\t{}'.format(repr_sentence(sentence, c_position)))
			print (u'\nCTX(F = {}, CIBLE_INCLUSE = {}) : '.format(F, CIBLE_INCLUSE))
			print (u'{:>32s} {:>6s} {:>32s}'.format(u'Token',u'POS',u'Lemme'))
			for ctx in CTX :
				print (u'{:>32s} {:>6s} {:>32s}'.format(*ctx))

			if candidats and scores:
				print (u'\n{:>20s} {:>17s}'.format(u'SUBSTITUANTS',u'SCORES'))
				for cible, score in zip(candidats, scores) : print (u'{:>20s} {:>16.15f}'.format(cible, score))
				print(u'\n')

				# sortie fichier formaté
				fout.write (u'{}.{} {} :: '.format(c, c_pos,id))
				for i, cible in enumerate(candidats) :
					fout.write(u'{}'.format(rm_pos(cible)))
					if i < len(candidats) - 1 : fout.write(u' ; ')
				fout.write(u'\n')

	# fermeture du fichier de sortie qui conient les réponses à évaluer
	if fout : fout.flush(); fout.close()

	# function d'évaluation provenant des "utilisateurs" de SemDis2014
	# le résultat d'évaluation reste très mauvais -> il y a eput-être des bogues, etc.
	# il est intéressant de lancer des tests pour voir avec quels paramètres (F,CIBLE_INCLUSES)
	# on a le meilleur résultat
	goldfile = 'semdis2014_lexsub_gold.txt'
	testFile = args.outfile
	s = semdis_eval.SemdisEvaluation(goldfile)
	s.evaluate(testFile, metric = 'all', normalize = True)
