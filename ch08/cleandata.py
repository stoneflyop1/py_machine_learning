import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # remove html markups
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
    return text

if __name__ == '__main__':
    text = "</a>This :) is :( a test :-)!"
    print('#'*20 + 'original text:')
    print(text)
    print('#'*20 + 'cleaned text:')
    print(preprocessor("</a>This :) is :( a test :-)!"))