import findspark
import pyspark
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer 
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd

# Bước 1: Chuyển đổi dataframe sang dữ liệu pyspark
'''
    parameter: df-thể hiện dữ liệu dataframe
    return: trả về dữ liệu kiểu định dạng pyspark
'''
def upload_dataframe(df):
    
    findspark.init()
    spark = SparkSession.builder.appName('text_context').getOrCreate()

    spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()
    #Create PySpark DataFrame from Pandas
    data_pyspark=spark.createDataFrame(df) 
    return data_pyspark
  


# Bước 2: Chuyển dữ liệu chữ vào trong pipeline
'''
    parameter:
        - df_pys: dữ liệu thể hiện là pyspark
        - lst_inputcol: dữ liệu cột text
    return: Trả về dữ liệu mã hóa để thực hiện vào model
'''
def pipeline_psy(df_pys, lst_inputCol):
    regextokenizer = RegexTokenizer(inputCol=lst_inputCol, outputCol='token_text', pattern='\\W') 
    stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens') 
    cv = CountVectorizer(inputCol='stop_tokens', outputCol='cv_text') 
    idf = IDF(inputCol='cv_text', outputCol='tf_idf_text')

    pre_pipe = Pipeline(stages=[regextokenizer, stopremove, cv, idf])
    cleaner = pre_pipe.fit(df_pys)
    pre_data = cleaner.transform(df_pys)
    return pre_data


# Bước 3: Chon k tốt nhất trong model kmean
'''
    parameter:
        - pre_data: lấy từ pipeline
        - k chọn từ k_from tới k_to
    return k_list, wsse_list, sihoutte_list các chỉ số
'''
def choose_k_in_kmeans(pre_data, k_from=2, k_to=5):
    # Trains a k-means model
    k_list = []
    wsse_list = []
    sihoutte_list = []

    wsse_str = ''
    sil_str = ''

    for k in range (k_from,k_to):
        k_list.append(k)
        kmeans = KMeans(featuresCol='tf_idf_text', k=k)
        model = kmeans.fit(pre_data)
        # wsse
        wsse = model.summary.trainingCost
        wsse_list.append(wsse)
        # sihoutte
        predictions = model.transform(pre_data)
        # Evaluate clustering by computing Sihoutte score
        evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='cv_text')
        sihoutte = evaluator.evaluate(predictions)
        sihoutte_list.append(sihoutte)
        
        wsse_str = wsse_str + 'with k =' + str(k) + '- Set sum of Squared Errors'+ str(wsse)+'\n'
        
        sil_str = sil_str + 'With k=' + str(k) + "- Sihoutte = "+str(sihoutte)+'\n'

        print('wsse_str:  ',wsse_str, 'sil_str: ',sil_str)
    return k_list, wsse_list, sihoutte_list

# Bước 4: thực hiện visualizaion để chon k
'''
    parameter:
        - k_list: list k
        - sihoutte_list: kết quả hiệu qua khi chọn nhóm
    return: hình ảnh dữ liệu
'''
def visualization_k (k_list, sihoutte_list):
    k_list.sort(reverse=True)
    result = (
    plt.plot(k_list, sihoutte_list),
    plt.show()
    )
    return result

# Bước 5: Thực hiện model thuật toán kmean
'''
    parameter:
        - pre_data: lấy ở pipeline
        - k: chọn k phù hợp
        - Giá trị mặc định
    return: wsse, sihouette, lst,pre, df_psy (bảng dự đoán theo cụm)
'''
def model_kmeans(pre_data, k, featuresCol = 'tf_idf_text', featuresCol2= 'cv_text',predictionCol ='prediction'):
    # Train a k-means model
    kmeans = KMeans(featuresCol = featuresCol, k=k)
    model = kmeans.fit(pre_data)
    # Evaluate clustering by computing Within Set Sum of Squared Errors
    wsse = model.summary.trainingCost
    print('Within Set Sum of Squared Errors = ' + str(wsse))
    # Sihoutte
    predictions  = model.transform(pre_data)
    # Evaluate clustering by computing Sihoutte score
    evaluator = ClusteringEvaluator(predictionCol=predictionCol, featuresCol=featuresCol2)
    sihouette = evaluator.evaluate(predictions)
    print('Sihoutte = ' + str(sihouette))
    # Show the resutl
    results = model.clusterCenters()
    print('Cluster Centers: ')
    lst = []
    for result in results:
        lst.append(lst)
    
    pre = model.transform(pre_data)
    df_psy = pre.groupBy('prediction').count().show()

    return wsse, sihouette, lst,pre, df_psy

# Bước 6: chọn data từ dữ liệu pyspark chuyển sang dataframe
'''
    parameter: 
        - pre: dùng dataframe
        - text: nội dung content
    return: Trả về dataFrame mong muốn
'''
def choose_select_wordText(pre, text, tf_idf_text='tf_idf_text',cv_text='cv_text', prediction='prediction'):
    data_result = pre.select(tf_idf_text, cv_text, text, prediction)
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    stopwords = set(STOPWORDS)
    wordText = data_result.select(text,prediction).toPandas()
    return wordText

# Bước 7: Hiện thị dữ liệu dang array
'''
    parameter:
        - wordtext: dữ liệu là datafram
        - text: chọn cột muốn hiển thị
    return
        - wordcloud trong group class
''' 
def display_word_kmeans(wordText, text_, prediction='prediction', k_from=0, k_to=3, max_words=100):
    # lấy top 10 Keyword mỗi cụm theo thuật toán Kmean
    stopwords = set(STOPWORDS)
    for i in range(k_from,k_to):
        print('Keyword Kmean Class:',i)
        text = " ".join(review for review in wordText[text_][wordText[prediction]==i])
        wordcloud = WordCloud(stopwords=stopwords, max_words=max_words).generate(text)
        #print(wordcloud.words_.keys())
        #print(type(wordcloud)
        #newlist = list()
        newlist = []
        for i in wordcloud.words_.keys():
            newlist.append(i)
    dtf_X = pd.DataFrame(newlist)
    return dtf_X


# Bước 8: Hiện thì những từ xuất hiện nhiều trong bài toán
'''
    parameter:
        - wordtext: dữ liệu là datafram
        - text: chọn cột muốn hiển thị
    return
        - wordcloud trong group class
'''
def visualization_wordcloud(wordText, text_, prediction='prediction', k_from=0, k_to=3, max_words=100):
    stopwords = set(STOPWORDS)
    # Trực quan hóa theo kết quả phân nhóm từ Kmean
    for i in range(k_from,k_to):
        print('Keyword Kmean Class:',i)
        text = " ".join(review for review in wordText[text_][wordText[prediction]==i])
        wordcloud = WordCloud(stopwords=stopwords, max_words=max_words,background_color='white').generate(text)
        resutl =(
            plt.figure(figsize=(15,5)),
            plt.imshow(wordcloud, interpolation='bilinear'),
            plt.axis("off"),
            plt.show()
        )
    return resutl
        