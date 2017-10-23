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

for c in 0 1
do
for f in 0 1 2 3 -1
do
for method in 0 1 2 3
do
echo '\tmethod :' $method
testfile='test_method_'$method'_cible_incluse_'$c'_F_'$f
echo '\ttestfile :' $testfile

if [ $c -eq 0 ]
then
if [ $f -eq 0 ]
then
continue
fi
fi

if [ $c -eq 1 ]
then
	python main.py $infile $resfile $resfolder2 $testfile -r $method -s $oversample -f $f -c
else
	python main.py $infile $resfile $resfolder2 $testfile -r $method -s $oversample -f $f
fi
python semdis_eval.py -g $goldfile -t $testfile
done
done
done
