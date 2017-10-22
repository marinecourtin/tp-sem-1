# coding:utf-8

from semdis_eval import *
from lexsub import *

import collections

if __name__ == "__main__" :

	model = word2vec.load('frWac_postag_no_phrase_700_skip_cut50.bin')
	eval = SemdisEvaluation('semdis2014_lexsub_gold.txt')
	gdic, sAdic = eval.parseGoldStandard()

	# gold : regrouper les substituatns par leur mot cible sans distinction d'identifiant
	d = collections.defaultdict(set)
	for k in gdic.keys() :
		target = k[0]
		gold_set = set(gdic[k].keys())
		d[target] = d.setdefault(target, set()).union(gold_set)

	"""
	for k in d.keys() :
		for n,gs in enumerate(list(d[k])) :
			print k,n,gs
	"""

	# eval : récuperer le vocabulaire de W2V
	nb2=0
	loss_targets=[]
	for k in d.keys() :
		if k.replace('.','_') in model.vocab : nb2+=1
		else : loss_targets.append(k)
	print ('Couverture de mots cibles pour le vocabulaire de W2V par rapport à la réponse GOLD = {}'.format( nb2 / float(len(d.keys())) ))
	print ('Autrement dit, les mots cibles non couverts par le vocabulaire de W2V = ')
	for i,x in enumerate(loss_targets) : print (i,x)

	print ('Résultat de la couverture lexicale par mot cible (tous les contextes confondus) pour l\'ensemble du vocabulaire de W2V par rapport à la réponse GOLD')
	for k in d.keys() :
		target = k
		nb2 = 0
		pos = k.split('.')[-1]
		for x in d[k] :
			if x + '_' + pos in model.vocab : nb2+=1
		cov = nb2 / float(len(d[k]))
		print (u'{:>16s} = {:>4.2f}'.format(target, cov))



