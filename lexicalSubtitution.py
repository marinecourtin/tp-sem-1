#coding:utf-8

import numpy, argparse

# note 1 : la méthode de repérage du mot cible à été modifiée. Dans cette version,
#          elle compte sur la position fournie pour le mot cible dans le fichier d'entrée
# note 2 : le fenêtrage a été re-implémtné avec la function absolue, i.e. abs()
#          pour améiorer la lisibilité

# todo : les lemmes par MElt peuvent contenir des ambiguïtés, par exemple :
#        "devoir|durer" -> à résoudre avant de passer à Word2vec

# réparation pour quelques cas de lemmatisation inattendue
lemmes_fix = {'compris' : 'comprendre'}
# liste des catégories morpho-syntaxiques considérées comme ensemble des mots pleins
cats_full = ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV']

def windowing (lst, position_center, single_side_width, center_included) :

	"""
	Cette function implémente un fenêtrage paramétrable sur une liste de données.
	Le contexte capturé est limitable à single_side_width tokens à gauche,
	single_side_width tokens à droite. Le mot cible est inclu si
	center_included est vrai.

	args :
	lst (list) : liste de données à fenêtrer
	position_center (int) : l'indice correspondant à la position du mot cible
	single_side_width (int) : le contexte est limitable à
	[position_center - single_side_width, position_center + single_side_width]
	center_included (bool) : true pour includre le mot cible dans le contexte; false, sinon

	return (list) : liste de données échantillonnées
	"""

	ret = []
	for position_cursor, elt in enumerate(lst) :
		dist = (position_cursor - position_center)
		if (dist == 0 and center_included) or \
		   (dist > 0 and abs(dist) <= single_side_width) or \
		   (single_side_width < 0) :
			ret.append(elt)
	return ret

def stopword_removal_by_cat (lst, position_center, cats_full) :

	"""
	Cette fonction enlève dans la liste d'entrée "lst" tous les mots 
	appartenant à des catégories morpho-syntaxiques non-renseignées 
	par la liste cats_full.
	lst (list) : liste des données d'entrée
	position_center (int) : l'indice correspondant à la position du mot cible
	cats_full (list(str)) : liste contenant les étiquettes de
	catégorie mophosyntaxique des mots pleins (jeu d'étiquettes par MElt)
	return (list, int) : le result de la suppression, la position du mot cible dans resultat
	"""

	ret = []
	new_position_cursor = 0
	new_position_center = -1
	for position_cursor, elt in enumerate(lst) :
		pos = elt[1]
		if position_cursor == position_center :
			new_position_center = new_position_cursor
		if pos in cats_full :
			ret.append(elt)
			new_position_cursor += 1
	return ret, new_position_center

def makeDico(inputFile, contextWindow = 3, includeTarget = True) :

	"""
	input :
		file containing target tokens and their data (context, id...)
	output : un dictionnaire suivant le format
		{int(instance):{'cible':token, 'lemme': lemme, 'pos':pos, 'position_phrase':int, 'phrase':[(tok, pos), (tok, pos)..]}}
		la liste des listes de contextes
	"""

	with open(inputFile) as fid :
		dico = {}
		contextes = []
		for line in fid :

			infos = line.split("\t")
			id_instance     = int(infos[0])
			lemme           = infos[1]
			pos             = infos[2]
			position_phrase = int(infos[3]) - 1 #index dans liste, du mot cible
			phrase_in       = infos[4]

			phrase = []
			for item in phrase_in.split() :
				tag, lem = item.split("/")[1 : ] #on prend le lemme et non le token
				phrase.append((lem, tag))

			# manuelly fix
			if lemme in lemmes_fix : # pb dans la lemmatisation proposee
				lemme = lemmes_fix[lemme]

			# on filtre le contexte selon la contextWindow
			[phrase_filtre, position_phrase_new] = stopword_removal_by_cat (phrase, position_phrase, cats_full)
			CTX = windowing (phrase_filtre, position_phrase_new, contextWindow, includeTarget)
			contexte = [elt[0] for elt in CTX]
			contextes.append(contexte)

			# un dico de dico
			dico[id_instance] = {'lemme'           : lemme, \
					     'pos'             : pos  , \
                                             'position_phrase' : position_phrase, \
                                             'phrase'          : CTX, \
                                             'contexte'        : contexte}
	return dico, contextes


if __name__ == '__main__' :

	parser = argparse.ArgumentParser(description='À ajouter')
	parser.add_argument('infile', help = 'file containing target tokens and their data (context, id...)', type = str)
	args = parser.parse_args()

	[dico, contextes] = makeDico(args.infile)
	print(contextes)
