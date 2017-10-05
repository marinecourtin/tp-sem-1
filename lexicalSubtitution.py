import numpy

def makeDico(inputFile, contextWindow=3, includeTarget=True):
	"""
	input :
		file containing target tokens and their data (context, id...)
	output : un dictionnaire suivant le format
		{int(instance):{'cible':token, 'lemme': lemme, 'pos':pos, 'position_phrase':int, 'phrase':[(tok, pos), (tok, pos)..]}}
		la liste des listes de contextes
	"""
	data = open(inputFile).read()
	targets = [line for line in data.split("\n") if line != ""]
	dico= {}
	contextes=[]
	for cible in targets:
		infos = cible.split("\t")
		id_instance = int(infos[0])
		lemme = infos[1]
		pos = infos[2]
		position_phrase = int(infos[3])-1 #index dans liste
		phrase=[]
		for item in infos[4].split(" "):
			infos = item.split("/")
			tag = infos[1]
			mot = infos[2] #on prend le lemme et non le token
			phrase.append((mot, tag))
		token = phrase[position_phrase][0] # au cas ou on en ait besoin plus tard
		if lemme == "compris": # pb dans la lemmatisation proposee
		  	lemme = "comprendre"
		pos_MELT = phrase[position_phrase][1]

		# on filtre le contexte selon la contextWindow
		phrase_filtre = [(tok, pos) for (tok, pos) in phrase if pos in ['ADJ', 'NC', 'NPP', 'V', 'VINF', 'VIMP', 'VPP', 'ADV']]
		if contextWindow==-1: #valeur infinie, on prend tous les mots pleins
			CTX = phrase_filtre
		else:
			count=0
			for elt in phrase_filtre:
				count+=1
				if elt == (token, pos_MELT):
					position_filtre=count #on recupere le rang du token dans le contexte
			borne_inf = max(position_filtre-contextWindow-1, 0)
			borne_sup = min(position_filtre+contextWindow, len(phrase_filtre))
			CTX = phrase_filtre[borne_inf:borne_sup]

			if includeTarget ==False:
				CTX = [(mot, tag) for (mot, tag) in CTX if (mot, tag) != (token, pos_MELT)]

			contexte=[]
			for i in range(len(CTX)-1):
				contexte.append(CTX[i][0])
		contextes.append(contexte)
		dico[id_instance]={'lemme':lemme, 'pos':pos, 'position_phrase':position_phrase, 'phrase':CTX, 'contexte':contexte}
	return dico, contextes
