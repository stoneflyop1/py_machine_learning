def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
# import nltk
# nltk.download('stopwords') # will download to ~/nltk_data/
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer_porter(text, ignore_stop=True):
    if ignore_stop:
        return [porter.stem(word) for word in text.split() if word not in stop]
    else:
        return [porter.stem(word) for word in text.split()]


if __name__ == '__main__':
    text = 'runners like running and thus they run'
    print('#'*20 + ' original text...')
    print(text)
    print('#'*20 + ' tokenizer by whitespace...')
    print(tokenizer(text))
    print('#'*20 + ' tokenizer by porter...')
    print('#'*15 + ' has stop...')
    print(tokenizer_porter(text, False))
    print('#'*15 + ' ignore stop...')
    print(tokenizer_porter(text))