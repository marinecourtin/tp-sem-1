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
from lexsub import *

# VARIABLES GLOBALES
n_candidats = 10 # max. selon la compagne semdis 2014

if __name__ == '__main__' :

      parser = argparse.ArgumentParser(description='Analyse sémantique TP-1 : substitution lexicale')
      parser.add_argument('infile', type=str, help='fichier de données de test : mots cibles et leurs contextes')
      parser.add_argument('resfile1', type=str, help='fichier des vecteurs de mot pré-générés via word2vec')
      parser.add_argument('resfolder2', type=str, help='dossier contenant les 4 fichiers composant le thésaurus FREDIST')
      parser.add_argument('outfile', type=str, help='fichier de réponse en sortie')
      parser.add_argument('-c','--cible_incluse', help='inclure le mot cible dans le contexte', action = "store_true")
      parser.add_argument('-f','--f_width', type=int, help='taille de la fenêtre contextuelle',default = 3)
      parser.add_argument('-r','--method', type=int, help="le type de ressource lexciale employée : 0 pour FRWAC (par défault), 1 pour FRDIC, 2 pour HYBRIDE, 3 pour FRWAC avec une nouvelle méthode", default = 0) #ajout d'un 3eme mode d'utilisation des ressources
      parser.add_argument('-s','--oversample', type=int, help='le ratio de suréchantillonnage',default = 1)
      parser.add_argument('-v','--verbose', help='voir des détails en temps réel', action="store_true")
      args = parser.parse_args()

      if args.verbose :
           print (args)


      method = args.method
      OVER_SAMPLING = args.oversample
      # chargement des ressources lexicales
      model = word2vec.load(args.resfile1)
      dicos = get_dicos_from_thesaurus(resfolder = args.resfolder2)
      print('initialisation du thésaurus en dictionnaires !')

      with codecs.open(args.infile, encoding = 'utf-8') as fin :
            with codecs.open(args.outfile, 'w', encoding = 'utf-8') as fout :

                  t1 = time.time()
                  for line in fin :

                        # lecture d'un échantillon de donnée et les segmenter en éléments
                        id, c, c_pos, c_position, sentence = line.split(u'\t')
                        tokens = [t.split(u'/') for t in sentence.split()]

                        # préparation et nettoyage du contexte plein
                        c_position_new, tokens_full = \
                              rm_stopword_from_tokens(tokens, cat_full, c_position)
                        overwindowing,CTX = \
                              windowing (tokens_full, c_position_new, args.f_width, args.cible_incluse)
                        CTX = clean_ctx(CTX)


                        # géneration des substituts potentiels
                        if method == 1:
                              candidats = generateSubstitutes(dicos, c, c_pos, n_candidats)
                        elif method == 2:
                              # on prend les 100 premiers résultats dans FREDIST comme substituts potentiels
                              # nb de substituts choisi arbitraire
                              candidats = generateSubstitutes(dicos, c, c_pos, n = 100)
                              try: #équivalent à la condition 'if candidats == None :'
                                    candidats = [candidat + u"_" + c_pos for candidat in candidats]
                              except TypeError: #cible pas dans FREDIST, on retombe sur r = 0
                                    candidats, scores = \
                                    generateSubstitutes_w2v(model, c, c_pos, OVER_SAMPLING * n_candidats)

                              Z = continous_bag_words(model, CTX)
                              candidats, scores = sort_response(model, candidats, Z)
                              candidats = candidats[0 : n_candidats]
                              print (len(candidats))
                              print (candidats)
                        elif method == 3 :

                              # nouvelle proposition basée sur les ressources word2vec
                              candidats, scores = generateSubstitutes_w2v(model, c, c_pos, OVER_SAMPLING * n_candidats)
                              candidats, scores = sel_cand_new (model, candidats, CTX, c, c_pos, n = 10)

                        else : # i.e. args.method == 0

                              candidats, scores = \
                              generateSubstitutes_w2v(model, c, c_pos, OVER_SAMPLING * n_candidats)

                              # ordonnancement de la liste des substituts proposés par le contexte
                              Z = continous_bag_words(model, CTX)
                              candidats, scores = sort_response(model, candidats, Z)
                              candidats = candidats[0 : n_candidats]


                        # sorties
                        if args.verbose :
                              show_infobox (id, c, c_pos, c_position, sentence, args.f_width, args.cible_incluse, CTX, method == 1)
                              print('candidats de substitut proposés : ')
                              #if candidats :
                              for i, cand in enumerate(candidats) :
                                  if cand : print(u'\t{} : {}'.format(i, cand))
                        if candidats :
                              export_substituants (id, c, c_pos, candidats, fout)

                  # évaluation
                  if method != 1 :
                      print (u'OVERSAMPLING RATIO : {}'.format(OVER_SAMPLING))
                  print (u'(MÉTHODE, F, CIBLE_INCLUSE) = ({}, {}, {})'.format(method, args.f_width, args.cible_incluse))

            print (u"Temps écoulée : {}".format(get_duration(t1_secs = t1, t2_secs = time.time())))
