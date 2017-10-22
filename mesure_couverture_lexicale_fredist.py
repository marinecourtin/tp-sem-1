# coding:utf-8

from semdis_eval import *
from lexsub import *

import collections

if __name__ == "__main__" :

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

	# eval : récuperer la liste des candidats provenant de FREDIST
	err_cnt = 0
	loss_targets = []
	d2 = collections.defaultdict(set)
	for target in d.keys() :
		c, c_pos = target.split('.')
		can = (generateSubstitutes(c, c_pos))
		if can :
			d2[target] = d2.setdefault(target, set()).union(set(can))
		else:
			err_cnt += 1
			loss_targets.append(target)
			#print ('err {} : {} do not existe in FREDIST, auncuns voisins ne seront proposés !!!'.format(err_cnt ,target))

	"""
        for k in d2.keys() :
                for n,gs in enumerate(list(d2[k])) :
                        print k,n,gs
	"""

	print ('Couverture de mots cibles pour FREDIC par rapport à la réponse GOLD = {}'.format( len(d2.keys()) / float(len(d.keys())) ))
	print ('Autrement dit, les mots cibles non-couverts sont')
	for i,l in enumerate(loss_targets) : print (i,l)

	# calculer intersection (gold, eval) étant un sous-ensemble de gold qu'on compare à gold en affichage
	print ('Résultat de la couverture lexicale par mot cible (tous les contextes confondus) pour FREDIST par rapport à la réponse GOLD')
	for k in d2.keys() :
		target = k
		cov = len(d[k] & d2[k]) / float(len(d[k]))
		print (u'{:>16s} = {:>4.2f}'.format(target, cov))


	# calculer |intersection (gold, eval)| / |gold|


