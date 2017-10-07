#! /usr/bin/env python3
#coding:utf-8

import argparse, word2vec, sys, numpy

cat_ok = ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV'] # reprise de la catégorie de Marine
catMelt2catFRWAK = {'NPP'  : 'n', \
		    'NC'   : 'n', \
                    'ADJ'  : 'a', \
                    'ADV'  : 'adv',\
                    'V'    : 'v',\
                    'VINF' : 'v',\
                    'VIMP' : 'v',\
                    'VPP'  : 'v'}

def combine (lemme, pos) :
	ret = lemme + u'_' + catMelt2catFRWAK[pos]
	return ret

def generate_response(w2v_model, vec, pos_desired, n = 10):

	"""
	similarité de cosinus adaptée de la fonction cosine provenant de
	https://github.com/danielfrg/word2vec/blob/master/word2vec/wordvectors.py
	"""

	# normalisation de vecteur 'vec': v = v / |v|, si |v| > 0
	if numpy.linalg.norm(vec) != 0 : vec = vec / numpy.linalg.norm(vec)

	metrics = numpy.dot(w2v_model.vectors, vec)
	best = numpy.argsort(metrics)[::-1][1:]

	cnt = 0
	candidats = []
	scores = []
	for cursor, elt in enumerate(best) :
		word_pos = w2v_model.vocab[elt]
		pos = word_pos.split('_')[-1]
		if pos == pos_desired :
			candidats.append(w2v_model.vocab[elt])
			scores.append(metrics[elt])
			cnt += 1
			if cnt == n : break

	return candidats, scores

if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
	parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
	args = parser.parse_args()

	# ressources
	w2v = 'frWac_postag_no_phrase_700_skip_cut50.bin'
	model = word2vec.load(w2v)

	# hyperparamètres à varirer par la suite
	CIBLE_INCLUSE = True
	F = 3

	with open(args.infile) as f :
		for line in f :
			id, c, c_pos, c_position, sentence = line.split('\t')
			tokens = [seg.split('/') for seg in sentence.split()]
			# filtrage des mots vides par catégorie morpho-syntaxique
			tokens_filtered = []
			j = 0
			for i, seg in enumerate(tokens) :
				token, pos, lemme = seg
				if i + 1 == int(c_position) : c_position_new = j
				if pos in cat_ok :
					tokens_filtered.append(seg) ; j += 1
			# construction du contexte par méthode de fenêtrage
			CTX = []
			for i, seg in enumerate(tokens_filtered) :
				if F < 0 :
					CTX.append(seg)
				elif i == c_position_new :
					if CIBLE_INCLUSE : CTX.append(seg)
				elif abs(i - c_position_new) <= F :
					CTX.append(seg)
			# vectorisation des mots en contexte
			Z = []
			for ctx in CTX :
				token, pos, lemme = ctx
				lemme_pos = combine(lemme.lower(), pos)
				try :
					if not Z : Z = model[lemme_pos]
					else     : Z += model[lemme_pos]
				except :
					# fixme : des ambiguïtés dans la forme lemmatisée
					# comme vivre|voir dans vit/V/vivre|voir à résoudre
					# afin de mieux profiter des ressources lexicales
					continue

			candidats, scores = generate_response(model, Z, catMelt2catFRWAK[pos])
			print ('instance id : {}'.format(id))
			print ('target token : cible {}'.format(c))
			print ('CTX(F = {}, CIBLE_INCLUSE = {}) : '.format(F, CIBLE_INCLUSE))
			for ctx in CTX : print (ctx)
			print ("{:<20s} {:<16s}".format('substituants','scores'))
			for c, s in zip(candidats, scores) : print ('{:>20s} {:>16.15f}'.format(c, s))
			print('\n')

