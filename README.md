## Evaluate English and Dutch Word Vectors on Lexical Norm Prediction

Usage:

To evaluate e.g. a vector set ```wiki.en.vec``` on it's ability to predict norms of valence, arousal, dominance, concreteness, and aoa in English, run:

```
python predict-norms.py -v wiki.en.vec -n norms-vadc-aoa-en.csv
```

To do the same on a Dutch vector set e.g. ```wiki.nl.vec```, run:

```
python predict-norms.py -v wiki.nl.vec -n norms-vadc-aoa-nl.csv
```

The normsets in ```norms-vadc-aoa-en.csv``` and ```norms-vadc-aoa-nl.csv``` are compiled from:

valence, arousal, dominance (English)  
<https://link.springer.com/article/10.3758%2Fs13428-012-0314-x>

aoa 50k (English)  
<http://crr.ugent.be/archives/806>

valence, arousal, dominance, aoa (Dutch)  
<https://link.springer.com/article/10.3758%2Fs13428-012-0243-8>

concreteness 40k (English)  
<http://crr.ugent.be/archives/1330>

concreteness 50k (Dutch)  
<http://crr.ugent.be/archives/1602>

