#! /bin/sh

infile="lexsubfr_semdis2014_test.id_melt" # fichier de données de test : mots cibles et leurs contextes
resfile="frWac_postag_no_phrase_700_skip_cut50.bin" # fichier des vecteurs de mot pré-générés via word2vec
resfolder2="the"
testfile="test.txt"
goldfile="semdis2014_lexsub_gold.txt" # fichier de réponse gold
oversample=3
method=2 # 0 W2C, 1 FRDIC, 2. HYBRIDE 3. new for W2V
echo "Generation des réponses ..."
python main.py $infile $resfile $resfolder2 $testfile -r $method -s $oversample -v
echo "Evaluation des réponses"
python semdis_eval.py -g $goldfile -t $testfile
