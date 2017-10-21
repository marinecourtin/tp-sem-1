# coding:utf-8
import argparse, word2vec, sys, numpy, codecs

dico_lemme_pos_fix = {'compris_a':'comprendre_v'}
cat_full = ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV'] # POS pour les mots pleins

def rm_pos(lemme_pos) :
	if u'_' not in lemme_pos :
		return lemme_pos
	lemme_pos = lemme_pos.split(u'_')
	lemme = u'_'.join(lemme_pos[0 : len(lemme_pos) - 1])
	return lemme

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
	if not candidats : return None, None
	if Z is None : return candidats, [-1 for x in candidats]

	if numpy.linalg.norm(Z) != 0 : vec = Z / numpy.linalg.norm(Z)
	for candidat in candidats :
		score = numpy.dot(w2v_model[candidat], Z)
		scores_candidats.append([score, candidat])
	scores_candidats = sorted(scores_candidats, key=lambda x : x[0], reverse = True)
	if scores_candidats :
		scores    = [x[0] for x in scores_candidats]
		candidats = [x[1] for x in scores_candidats]
	return candidats, scores

def generateSubstitutes_w2v(w2v_model, c_lemme, c_pos, n = 10) :

	c_lemme_pos = c_lemme + u'_' + c_pos
	if c_lemme_pos in dico_lemme_pos_fix.keys() :
		c_lemme_pos = dico_lemme_pos_fix[c_lemme_pos]

	vec = w2v_model[c_lemme_pos]
	if vec is None : return None, None
	"""
	génération des substituts basée sur la similarité de cosinus,
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
	# les candidats doivent avoir la catégorie POS spécifié par 'pos_desired'
	cnt = 0
	candidats = []
	scores = []
	for cursor, i in enumerate(indexes_best) :
		word_pos = w2v_model.vocab[i]
		# ex: word_pos = 'intéresser_v' -> pos = 'v'
		pos = word_pos.split('_')[-1]
		if pos == c_pos :
			candidats.append(w2v_model.vocab[i])
			scores.append(metrics[i])
			cnt += 1
			if cnt == n : break # on ne prend que les n meilleurs

	return candidats, scores

def generateSubstitutes(c, c_pos, n=15): #surement optimisable
	"""
	sélectionne les candidats substituts à la cible. La fonction est basée sur
	l'utilisation de la ressource FREDIST disponible à l'adresse :
	https://gforge.inria.fr/projects/fredist/
	args : cible, pos de la cible, nombre de candidats
	output : liste des n candidats les plus similaires

	"""
	c_pos = c_pos.upper()
	with codecs.open("./thesauri-1.0/thesaurus_french_"+c_pos+'.txt', encoding = 'utf-8') as f :
		for line in f:
			line = line.split("\t")
			term, subs = (line[0].split("|")[1], line[1:][:n])
			if term == c:
				candidats = [sub.split("|")[1] for sub in subs]
				candidats = [can.split(":")[0] for can in candidats]
				# print(term, c, candidats)
		if not candidats:
			return None
	return candidats

def rm_stopword_from_tokens(tokens, cat_full, c_position) :
	tokens_full = []
	j = 0
	for i, t in enumerate(tokens) :
		token, pos, lemme = t
		if i + 1 == int(c_position) :
			c_position_new = j
		if pos in cat_full :
			tokens_full.append(t)
			j += 1
	return c_position_new, tokens_full

def windowing (CTX_in, c_position, F, CIBLE_INCLUSE) :
	CTX = []
	for i, t in enumerate(CTX_in) :
		if F < 0 :
			CTX.append(t)
		elif i == c_position :
			if CIBLE_INCLUSE :
				CTX.append(t)
		elif abs(i - c_position) <= F :
			CTX.append(t)

	if i - c_position < F : over_windowing = True
	else : over_windowing = False
	return over_windowing, CTX

def clean_ctx (CTX) :
	# nettoyage du contexte plein
	CTX2 = []
	for ctx in CTX :
		token, pos, lemmes = ctx

		# suppression de '*' dans les lemmes
		lemmes = lemmes.replace(u'*',u'')

		# traiter l'ambiguiïté dans les lemmes
		if u'|' in lemmes :
			for lemme in lemmes.split(u'|') :
				if lemme :
					CTX2.append([token,pos,lemme])
		else :
			CTX2.append([token, pos, lemmes])
	return CTX2

def continous_bag_words (w2v_model, CTX) :
	Z = None
	for ctx in CTX :
		token, pos, lemme = ctx
		lemme_pos = lemme.lower() + u'_' + conv_pos(pos)
		try :
			if not Z :
				# initier "la somme vectorielle" Z par la valeur du premier vecteur visité
				# comme ça Z est typé comme un vecteur
				Z = w2v_model[lemme_pos]
			else :
				# une fois l'initialisation faite, on accumule des vecteurs par
				# l'addition vectorielle ou pointwise addition
				#
				# note: lorsque le symbole + et utilisé comme un opérateur binaire
				# dans, par exemple a + b, '+' est devenue une addition vectorielle
				# si a, b sont du type vectoriel et de la même taille (longueur)
				Z += w2v_model[lemme_pos]
		except :
			continue
	return Z


def export_substituants (id, c, c_pos, candidats, fidout) :
	fidout.write (u'{}.{} {} :: '.format(c, c_pos,id))
	for i, cible in enumerate(candidats) :
		fidout.write(u'{}'.format(rm_pos(cible)))
		if i < len(candidats) - 1 : fidout.write(u' ; ')
	fidout.write(u'\n')

def show_infobox (id, c, c_pos, c_position, sentence, F, CIBLE_INCLUSE, CTX):
	print (u'instance id : {}'.format(id))
	print (u'target token : {}'.format(c))
	print (u'target POS : {}'.format(c_pos))
	print (u'full sentence : \n\t{}'.format(repr_sentence(sentence, c_position)))
	print (u'\nCTX(F = {}, CIBLE_INCLUSE = {}) : '.format(F, CIBLE_INCLUSE))
	print (u'{:>32s} {:>6s} {:>32s}'.format(u'Token',u'POS',u'Lemme'))
	for ctx in CTX :
		print (u'{:>32s} {:>6s} {:>32s}'.format(*ctx))
