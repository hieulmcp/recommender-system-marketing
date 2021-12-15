import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
from underthesea import classify
from underthesea.pipeline import classification
from underthesea import word_tokenize

### 1.1.0. Load những file thể hiện nhưng từ trong tiếng việt hay sử dụng
'''
    emojicon.txt: File thể hiện những icon mặt cười, mặt buồn...
    teencode.txt: những từ viết tắt trong tiếng việt khi comment
    english-vnmese.txt: viết tiếng anh xen tiếng anh
    wrong-word.txt: Những từ vựng bị sai
    vietnamese-stopwords.txt thể hiện những từ lặp đi lặp lai nhiều và không có ý nghĩa
    Cần viết hàm để lấy dữ liệu những file này vào và có thể update
'''
##LOAD EMOJICON
def icon_to_vietnamese():
    file = open('data/emojicon.txt', 'r', encoding="utf8")
    emoji_lst = file.read().split('\n')
    emoji_dict = {}
    for line in emoji_lst:
        key, value = line.split('\t')
        emoji_dict[key] = str(value)
    file.close()
    return emoji_dict
    
#################
#LOAD TEENCODE
def teencode_vietnamese():
    file = open('data/teencode.txt', 'r', encoding="utf8")
    teen_lst = file.read().split('\n')
    teen_dict = {}
    for line in teen_lst:
        key, value = line.split('\t')
        teen_dict[key] = str(value)
    file.close()
    return teen_dict
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
def english_to_vietnamese():
    file = open('data/english-vnmese.txt', 'r', encoding="utf8")
    english_lst = file.read().split('\n')
    english_dict = {}
    for line in english_lst:
        key, value = line.split('\t')
        english_dict[key] = str(value)
    file.close()
    return english_dict
################
#LOAD wrong words
def wrong_word():
    file = open('data/wrong-word.txt', 'r', encoding="utf8")
    wrong_lst = file.read().split('\n')
    file.close()
    return wrong_lst
#################
#LOAD STOPWORDS
def stopwords_lst_vietnamese():
    file = open('data/vietnamese-stopwords.txt', 'r', encoding="utf8")
    stopwords_lst = file.read().split('\n')
    file.close()
    return stopwords_lst

### 1.1. Process_test: thể hiện những https://topdev.vn/blog/regex-la-gi/ => Điều chỉnh lại các tự vựng không cần hiết
### Bước 1: Tiền xử lý dữ liệu thô để cho đúng với dữ liệu và xóa bỏ 1 ít những từ không cần thiết
'''
    text: chuỗi text cần thực hiện
    emoji_dict: thể hiện những teen code mặt cười...
    teen_dict: những teen code của VN
    wrong_lst: Những từ sai

'''
def process_text(text, emoji_dict = icon_to_vietnamese(), teen_dict = teencode_vietnamese(), wrong_lst = wrong_word()):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '                    
    document = new_sentence  
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Bước 2: Chuẩn hoá Unicode tiếng Việt
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
'''
    Mục đích: trả về nhưng từ sai unicode
    return về 1 chuỗi text
'''
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


# có thể bổ sung thêm các từ: chẳng, chả...
def process_special_word(text, word="không"):
    new_text = ''
    text_lst = text.split()
    i= 0
    if word in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == word:
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document


def remove_stopword(text, stopwords = stopwords_lst_vietnamese()):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Standardlize vietnamese unicode
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

# Bước 3: Tokenize văn bản tiếng việt
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

# Bước 4: Remove Stopword
def remove_stopword(text, stopwords = stopwords_lst_vietnamese()):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

#############################################################################################################################################################

#############################################################################################################################################################


# Bước 1: Phân tách chuổi tiếng việt
def process_text_step1(text):
    document = text.lower()
    specialCharacter = ['^','<','>','{','}','""','/','|',';',':','.',',','~','!', \
            '?','@','#','$','%','=','&','*','(',')','\\','[','¿','§','«',\
            '»','ω','⊙','¤','°','℃','℉','€','¥','£','¢','¡','®','©','0','-','9','_','+',']','*','$','--','“', '’', '--','?']
    
    document = document.replace("“",'')
    document = regex.sub(r'\.+', "", document)
    document = regex.sub(r'https?://(www.)?\w+\.\w+(/\w+)*/?', "", document)
    document = regex.sub(r'[%s]' % re.escape(string.punctuation), "", document)
    document = regex.sub(r'@(\w+)', "", document)
    document = regex.sub(r'(\\d+)' , "", document)
    document = regex.sub(r'[0-9]' , "", document)
    for i in specialCharacter:
        document = document.replace(i,'')
    

    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))              
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
# Chuẩn hóa unicode tiếng việt
def loaddicchar_():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Bước 2: Chuẩn hoá Unicode tiếng Việt
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
'''
    Mục đích: trả về nhưng từ sai unicode
    return về 1 chuỗi text
'''
def covert_unicode_(txt):
    dicchar = loaddicchar_()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

# Xử lý dữ liệu test theo underthesea
'''
    Parameter:
        - df: dataframe
        - lst_text: dữ liệu muốn thay đổi
    Return: Lấy dữ liệu về datafram đếm số từ và các cụm từ có hữu ích với bài toán

'''
def process_nlp_vietnamese(df,lst_text, columns_='word'):
    lst_txt_ = [] # Return về tách chuổi theo list [['bán', 'nhà', 'm', 'tân hiệp', 'hóc', 'môn', 'cách', 'đỗ', 'văn', 'dậy', 'm'], \
                                                    #['bán', 'nhà', 'nát', 'gần', 'mặt tiền', 'đường', 'hậu', 'giang', 'p', 'q', 'm']]
    for i in df.index:
        txt0 = df[lst_text][i]
        txt1 = process_text_step1(txt0)
        txt2 = covert_unicode_(txt1)
        #txt3 = remove_stopword(txt2)
        txt4 = word_tokenize(txt2, format="text") # Nối chuổi có nghĩa
        txt5 = txt4.strip()
        #txt4 = no_accent_vietnamese(txt3)
        #txt5 = word_tokenize(txt3) # Tách chuổi
        lst_txt_.append(txt5)
    wordcloud = pd.DataFrame(lst_txt_,columns=[columns_])
    return wordcloud

def txt_stop_word(df, lst_text,  columns_='word'):
    lst_txt_ = []
    for i in df.index:
        txt0 = df[lst_text][i]
        txt1 = remove_stopword(txt0)
        lst_txt_.append(txt1)
    wordcloud = pd.DataFrame(lst_txt_,columns=[columns_])
    return wordcloud

def groupby_word_txt(df,lst_text, columns_='word'):
    lst_txt_ = [] # Return về tách chuổi theo list [['bán', 'nhà', 'm', 'tân hiệp', 'hóc', 'môn', 'cách', 'đỗ', 'văn', 'dậy', 'm'], \
                                                    #['bán', 'nhà', 'nát', 'gần', 'mặt tiền', 'đường', 'hậu', 'giang', 'p', 'q', 'm']]
    for i in df.index:
        txt0 = df[lst_text][i]
        txt1 = process_text_step1(txt0)
        txt2 = covert_unicode_(txt1)
        txt3 = word_tokenize(txt2, format="text") # Nối chuổi có nghĩa
        #txt4 = no_accent_vietnamese(txt3)
        txt5 = word_tokenize(txt3) # Tách chuổi
        lst_txt_.append(txt5)
    words=[] # Return 1 list các chữ phân tách
    for m in range(0,len(lst_txt_)):
        for n in range(0,len(lst_txt_[m])):
            words.append(lst_txt_[m][n])
    # Chuyển dữ liệu qua dataframe
    df_Count = pd.DataFrame(words,columns=[columns_])
    df_Count['Num']= 1
    # Đếm dữ liệu theo từng từ
    df_GroupBy=df_Count.groupby(columns_).count()
    df_GroupBy.sort_values('Num',ascending=False,inplace=True)
    return df_GroupBy


#####################################################################################################
# Tách chuổi trong 1 câu 
#####################################################################################################
# Bước 1: Add thêm 1 feature thành nhiều feture theo mong muốn
'''
Parameter:
    - df: data theo kiểu dữ liệu ban đầu, sau khi xóa dữ liệu rác thông qua hàm change_vulue_category: dữ lại những thông tin cần thiết
    - lst_add_feature: những thuộc tính mới thêm
    - lst_feature: thuộc tính gốc
return dataframe => Thể hiện các mỗi quan hệ cần thiết
'''
def add_feature_data(df, lst_add_feature, lst_feature):
    for i in lst_add_feature:
        df[i] = '0 ' +i
        df.loc[df[lst_feature].str.contains(i)==True,i]=df[lst_feature].str[0:100]
    return df

# Bước 2.1: 
def delete_string(lst_add_feature, text, change_feature):
    lst_s = []
    for i in lst_add_feature:
        if i==change_feature:
            continue
        else:
            lst_s.append(i)
            for j in lst_s:
                    text = regex.sub(r'[0-9]+ '+j, "", text)
                    text = text.replace(",",'')
                    text = text.strip()
    return text

def delete_lst_string(df, lst_add_feature,lst_feature, change_feature, columns_=''):
    lst_txt_ = []
    df_new = add_feature_data(df= df, lst_add_feature = lst_add_feature, lst_feature = lst_feature)
    for i in df_new.index:
        txt0 = df_new[change_feature][i]
        txt1 = delete_string(lst_add_feature=lst_add_feature, text=txt0, change_feature=change_feature)
        lst_txt_.append(txt1)
        
    df_ = pd.DataFrame(lst_txt_,columns=[columns_])
    
    return df_



# Tạo function clean chữ Tiếng Việt

# Do dữ liệu đều là Tiếng Việt có dấu. Để tránh lỗi typing thì sẽ chuyển hết về Tiếng Việt không dấu
def no_accent_vietnamese(s):
    s = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', s)
    s = re.sub('[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'A', s)
    s = re.sub('[éèẻẽẹêếềểễệ]', 'e', s)
    s = re.sub('[ÉÈẺẼẸÊẾỀỂỄỆ]', 'E', s)
    s = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', s)
    s = re.sub('[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'O', s)
    s = re.sub('[íìỉĩị]', 'i', s)
    s = re.sub('[ÍÌỈĨỊ]', 'I', s)
    s = re.sub('[úùủũụưứừửữự]', 'u', s)
    s = re.sub('[ÚÙỦŨỤƯỨỪỬỮỰ]', 'U', s)
    s = re.sub('[ýỳỷỹỵ]', 'y', s)
    s = re.sub('[ÝỲỶỸỴ]', 'Y', s)
    s = re.sub('đ', 'd', s)
    s = re.sub('Đ', 'D', s)
    return s

def remove_stopword(text, stopwords = stopwords_lst_vietnamese()):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

############################################################################################################################################################

#############################################################################################################################################################
# Bước 1: Phân tách chuổi tiếng việt
def process_text_step_no_number(text):
    document = text.lower()
    specialCharacter = ['^','<','>','{','}','""','/','|',';',':','.',',','~','!', \
            '?','@','#','$','%','=','&','*','(',')','\\','[','¿','§','«',\
            '»','ω','⊙','¤','°','℃','℉','€','¥','£','¢','¡','®','©','0','-','9','_','+',']','*','$','--','“', '’', '--','  ']
    
    document = document.replace("“",'')
    document = regex.sub(r'\.+', "", document)
    document = regex.sub(r'https?://(www.)?\w+\.\w+(/\w+)*/?', "", document)
    document = regex.sub(r'[%s]' % re.escape(string.punctuation), "", document)
    document = regex.sub(r'@(\w+)', "", document)
    document = regex.sub(r'(\\d+)' , "", document)
    #document = regex.sub(r'[0-9]' , "", document)
    for i in specialCharacter:
        document = document.replace(i,'')
    
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))              
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Bước 2: Thay đổi NLP vietnamese
def process_nlp_vietnamese_no_number(df,lst_text, columns_='word'):
    lst_txt_ = [] # Return về tách chuổi theo list [['bán', 'nhà', 'm', 'tân hiệp', 'hóc', 'môn', 'cách', 'đỗ', 'văn', 'dậy', 'm'], \
                                                    #['bán', 'nhà', 'nát', 'gần', 'mặt tiền', 'đường', 'hậu', 'giang', 'p', 'q', 'm']]
    for i in df.index:
        txt0 = df[lst_text][i]
        txt0_ = txt0.strip()
        txt01_ = txt0_.strip()
        txt1 = process_text_step_no_number(txt01_)
        txt2 = covert_unicode_(txt1)
        #txt3 = remove_stopword(txt2)
        txt4 = word_tokenize(txt2, format="text") # Nối chuổi có nghĩa
        #txt4 = no_accent_vietnamese(txt3)
        #txt5 = word_tokenize(txt3) # Tách chuổi
        lst_txt_.append(txt4)
    wordcloud = pd.DataFrame(lst_txt_,columns=[columns_])
    return wordcloud

# Bước 3: Cách chọn và thực hiện cách tách chữ
def split_text(df, lst_slit, change_text, value_change):
    # Tách dữ liệu về giá trong chuỗi ký tự
    lst_gia = []
    for i in df[lst_slit]:
        document = i.lower()
        document_ = document.replace(change_text,value_change)
        document_1 = document_.strip()
        lst_gia.append(document_1)
    return lst_gia

# Bước 3: Cách chọn và thực hiện cách tách chữ
def split_text_one_feature(df, change_text, value_change):
    # Tách dữ liệu về giá trong chuỗi ký tự
    lst = []
    for i in df:
        document = i.lower()
        document_ = document.replace(change_text,value_change)
        document_1 = document_.strip()
        lst.append(document_1)
    return lst

# Bước 4: Chay gia tri thanh côt từ dữ liệu text
def dataFrame_text_to_list(df, lst_slit, change_text, value_change, text=" "):
    lst_gia = split_text(df, lst_slit, change_text, value_change)
    lst_amount = []
    lst_UoM_amount = []
    lst_price_house = []
    lst_UoM_price_house = []
    for i in lst_gia:
        lst_=i.split(text)
        lst_amount.append(lst_[0].replace(',','.'))
        lst_UoM_amount.append(lst_[1])
        lst_price_house.append(lst_[2].replace(',','.'))
        lst_UoM_price_house.append(lst_[3])
    wordcloud  = pd.DataFrame(
                            {'amount': lst_amount,
                            'UoM_amount': lst_UoM_amount,
                            'price_house': lst_price_house,
                            'UoM_price_house': lst_UoM_price_house
                            })
    wordcloud['amount'] = wordcloud['amount'].astype(float)
    wordcloud['price_house'] = wordcloud['price_house'].astype(float)
    return wordcloud

# Bước 5: Chuyen doi du lieu
def split_text_string(df, lst_slit, change_text, value_change, columns_ = ['Structure']):
    # Tách dữ liệu về giá trong chuỗi ký tự
    lst_gia = []
    for i in df[lst_slit]:
        document = i.lower()
        document = regex.sub(r'[0-9]' , "", document)
        #document = document.replace("'",'')
        document_ = document.replace(change_text,value_change)
        document_1 = document_.strip()
        #document_1 = document_1.str.split(',')
        lst_gia.append(document_1)
    df_new = pd.DataFrame(lst_gia, columns=columns_)
    return df_new

def category_dimension(df, lst_slit, change_text, value_change, columns_ =['Structure']):
    # Tách dữ liệu về giá trong chuỗi ký tự
    # Tách dữ liệu về giá trong chuỗi ký tự
    lst_gia = []
    for i in df[lst_slit]:
        document = i.lower()
        document = regex.sub(r'[0-9]' , "", document)
        #document = document.replace("'",'')
        document_ = document.replace(change_text,value_change)
        document_1 = document_.strip()
        #document_1 = document_1.str.split(',')
        lst_gia.append(document_1)
    mylist = list(dict.fromkeys(lst_gia))
    df_new = pd.DataFrame(mylist, columns=columns_)
    return df_new

def change_vulue_category(df, lst_slit, change_text, value_change, columns_ =['Structure']):
    lsts_ = []
    for i in df[lst_slit]:
        i = i.lower()
        i = i.strip()
        lst = i.replace(change_text,value_change)
        lsts_.append(lst)
    df_new = pd.DataFrame(lsts_, columns=columns_)
    return df_new

def change_vulue_category_lst(df, lst_slit, change_text, value_change,  change_text_=", "):
    lsts_ = []
    for i in df[lst_slit]:
        lst = i.replace(change_text,value_change)
        lsts_.append(lst)
    list1_joined = change_text_.join(lsts_)
    txt = list1_joined.split(change_text_)
    mylist = list(dict.fromkeys(txt))
    txt1 = change_text_.join(mylist)
    txt2 =txt1.split(change_text_)
    txt3 = list(dict.fromkeys(txt2))
    txts = []
    for i in txt3:
        txt4 = i.strip()
        txts.append(txt4)
    
    return txts

def add_feature(df, lst_text, lst_feature):
    for i in lst_text:
        df[i] = 0
        df.loc[df[lst_feature].str.contains(i)==True,i]=1
    return df


#############################################################################################################################################################

#############################################################################################################################################################
# B. XỬ LÝ CHUỖI THEO PYSPARK
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import langid
import string
import re

## C. XỬ LÝ DỮ LIỆU TEXT TIẾNG ANH
### 1.6. Xử lý dữ liệu text
##### Check to see if a row only contains whitespace
##### Kiểm tra và nhìn thấy dữ liểu có chứa khoảng trắng
def check_blanks(data_str):
    try:
        is_blank = str(data_str.isspace())
        return is_blank
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.7. Check ngôn ngữ tiếng anh
##### Determine whether the language of the text content is english or not: Use langid module to classify
##### the language to make sure we are applying the correct cleanup actions for English langid
##### Kiểm tra ngôn ngữ English
def check_lang(data_str):
    try:
        predict_lang = langid.classify(data_str)
        if predict_lang[1] >= .9:
            language = predict_lang[0]
        else:
            language = 'NA'
        return language
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.8. Remove features
##### Remove những dữ liệu có chứa những ký tự đặc biệt và các ký tự không cần thiết
def removeFeatures(dataStr):
    try:
        # compile regex
        url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
        punc_re = re.compile('[%s]' % re.escape(string.punctuation))
        num_re = re.compile('(\\d+)')
        mention_re = re.compile('@(\w+)')
        alpha_num_re = re.compile("[^A-Za-z0-9]") # Kiểu phủ định khác các chữ cái trên tron ^ là kiểu phủ đỉnh
        #dataStr = pd.Series(dataStr)
        # convert to lowercase
        dataStr = dataStr.lower()
        # remove hyperlinks
        dataStr = url_re.sub(' ', str(dataStr))
        # remove @mentions
        dataStr = mention_re.sub(' ', str(dataStr))
        # remove puncuation
        dataStr = punc_re.sub(' ', str(dataStr))
        # remove numeric 'words'
        dataStr = num_re.sub(' ', str(dataStr))
        # remove non a-z 0-9 characters and words shorter than 3 characters
        list_pos = 0
        cleaned_str = ''
        
        for word in dataStr.split():
            if list_pos == 0:
                if alpha_num_re.match(word) and len(word) > 2:
                    cleaned_str = word
                else:
                    cleaned_str = ' '
            else:
                if alpha_num_re.match(word) and len(word) > 2:
                    cleaned_str = cleaned_str + ' ' + word
                else:
                    cleaned_str += ' '
            list_pos += 1
        return cleaned_str
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.9. removes stop words
##### removes stop words, phải lập những từ xuất hiền nhiều không có ý nghĩa
def remove_stops(data_str):
    try:
        # expects a string
        stops = set(stopwords.words("english"))
        list_pos = 0
        cleaned_str = ''
        text = data_str.split()
        for word in text:
            if word not in stops:
            # rebuild cleaned_str
                if list_pos == 0:
                    cleaned_str = word
                else:
                    cleaned_str = cleaned_str + ' ' + word
                list_pos += 1
        return cleaned_str
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.10. Tagging text - Gắn thẻ vẵn bản
def tag_and_remove(data_str):
    try:
        cleaned_str = ' '
        # noun tags
        nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
        # adjectives
        jj_tags = ['JJ', 'JJR', 'JJS']
        # verbs
        vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        nltk_tags = nn_tags + jj_tags + vb_tags
        # break string into 'words'
        text = data_str.split()
        # tag the text and keep only those with the right tags
        tagged_text = pos_tag(text)
        for tagged_word in tagged_text:
            if tagged_word[1] in nltk_tags:
                cleaned_str += tagged_word[0] + ' '
        return cleaned_str
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.11. lemmatization
def lemmatize(data_str):
    try:
        # expects a string
        list_pos = 0
        cleaned_str = ''
        lmtzr = WordNetLemmatizer()
        text = data_str.split()
        tagged_words = pos_tag(text)
        for word in tagged_words:
            if 'v' in word[1].lower():
                lemma = lmtzr.lemmatize(word[0], pos='v')
            else:
                lemma = lmtzr.lemmatize(word[0], pos='n')
            if list_pos == 0:
                cleaned_str = lemma
            else:
                cleaned_str = cleaned_str + ' ' + lemma
            list_pos += 1
        return cleaned_str
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")
