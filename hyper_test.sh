#! /bin/sh

infile="lexsubfr_semdis2014_test.id_melt" # fichier de données de test : mots cibles et leurs contextes
resfile="frWac_postag_no_phrase_700_skip_cut50.bin" # fichier des vecteurs de mot pré-générés via word2vec
resfolder2="."
goldfile="semdis2014_lexsub_gold.txt" # fichier de réponse gold
oversample=3

#method=3 # 0 W2C, 1 FRDIC, 2. HYBRIDE 3. new for W2V
echo 'oversample :' $oversample
echo 'infile :' $infile
echo 'resfile (word2vec) :' $resfile
echo 'resfolder2 (thésaurus) :' $resfolder2
echo 'goldfile :' $goldfile

for method in 0 1 2 3
do
echo '\tmethod :' $method
testfile='test_method_'$method
echo '\ttestfile :' $testfile
python main.py $infile $resfile $resfolder2 $testfile -r $method -s $oversample
python semdis_eval.py -g $goldfile -t $testfile
done
