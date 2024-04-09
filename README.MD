<!-- import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from collections import Counter
import re
from heapq import nlargest
nltk.download('punkt')

class Clasa:

    with open('text.txt','r', encoding='utf-8') as file:
            text = file.read();       

    def nrPropozitii(text):
        propozitii = sent_tokenize(text)
        print("Numărul de propoziții în text:", len(propozitii))



Clasa.nrPropozitii();


propozitii = sent_tokenize(text)
print("Numărul de propoziții în text:", len(propozitii))

cuvinte = word_tokenize(text)
print("Numarul total de cuvinte este : ",  len(cuvinte))

semne_punctuatie = set(string.punctuation)
cuvinte_fara_puncte = [cuvant for cuvant in cuvinte if cuvant not in semne_punctuatie]
top_cuvinte = [cuvant for cuvant in cuvinte]
lungime_text_fara_punctuatie = len(cuvinte_fara_puncte)
print("Lungimea textului fără semne de punctuație:", lungime_text_fara_punctuatie)



cuvintee = [cuvant.lower() for cuvant in cuvinte]
frecventa_cuvinte = Counter(cuvintee)
top_5_cuvinte = frecventa_cuvinte.most_common(5)
print("Cele mai frecvente 5 cuvinte:")
for cuvant, frecventa in top_5_cuvinte:
    print(cuvant, ":", frecventa)



expresie_regulara_plural = re.compile(r'[s]')
cuvinte_plural = [cuvant.lower() for cuvant in cuvinte if expresie_regulara_plural.match(cuvant)]
numar_cuvinte_plural = len(cuvinte_plural)
print("Numărul total de cuvinte la plural în text:", numar_cuvinte_plural)


lungime_rezumat = int(len(propozitii) * 0.2)
rezumat = nlargest(lungime_rezumat, propozitii, key=len)
with open('rezumat.txt', 'w', encoding='utf-8') as rezumat_file:
    for propozitie in rezumat:
        rezumat_file.write(propozitie + '\n')

print("Rezumatul textului a fost scris în fișierul rezumat.txt.")
 -->
