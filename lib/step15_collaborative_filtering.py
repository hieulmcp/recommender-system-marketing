import findspark
findspark.init()
# import librariesf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# Create ALS model and fitting data
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
### Đánh giá
from pyspark.ml.evaluation import RegressionEvaluator



#### 1.2.1. Step 1: Create the SparkSession Object
def get_spark_session(title = 'Recommender Systems - Hệ thống đề xuất'):
    spark = SparkSession.builder.appName(title) \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.driver.python", "python3") \
        .getOrCreate()
    return spark

#### 1.2.2. Step 2: Read the Dataset: Chung ta khi load và đọc dữ liệu với Spark phải sử dụng dataframe.
def load_data(file_path = 'data_analysis/data_merge.csv', title = 'Recommender Systems - Hệ thống đề xuất'):
    spark = get_spark_session(title)
    df = spark.read.csv(file_path,inferSchema=True,header=True,  encoding='utf-8')
    return df

#### 1.2.2.1. Chọn dữ liệu
def select_data(df, lst=['']):
    data_sub = df.select(lst)
    return data_sub

#### 1.2.4. Step 4: Feature Engineering: Chúng tôi sẽ convert những column categorical sang numerical values using StringIndexex
def feature_engineering_stringIndexer(data_sub,inputCol="fea_name", outputCol="title_new"):
    stringIndexer = StringIndexer(inputCol=inputCol, outputCol=outputCol)
    model = stringIndexer.fit(data_sub)
    indexed = model.transform(data_sub)
    return indexed

#### 1.2.5. Step 5: Splitting the Dataset
#- Chung ta có thể so sánh dữ liệu xây dựng mô hình đề xuất, chung có thể chia dữ liệu theo dữ liệu training và test
#- Chung ta có thể chia trong khoảng 75 đến 25 chỉ số train mode và test accuracy
def random_split(data_sub, randoms = [0.75,0.25]):
    train,test=data_sub.randomSplit(randoms)
    return train,test

#### 1.2.6. Step 6: Build and Train Recommender Model
#- Chúng ta có thể import ALS thư viện Pyspark machine learning và xây dựng mô hình training dữ liệu
#- Có rất nhiều parameter nhưng có tuned cải thiện hiệu suất mô hình
#- Có 2 para quan trọng đó là: 
#    + nonnegative =‘True’: Nó không thêm negative ratings in recommendations
#    + coldStartStrategy=‘drop’ to prevent any NaN ratings predictions
def als_model(train, maxIter=15, userCol='fea_customer_id',itemCol='fea_item_id',ratingCol='fea_rating_y', nonnegative=True, coldStartStrategy="drop"):
    rec=ALS(maxIter=maxIter, userCol=userCol,itemCol=itemCol,ratingCol=ratingCol, nonnegative=nonnegative, coldStartStrategy=coldStartStrategy)
    rec_model=rec.fit(train)
    return rec_model

#### 1.2.7.1. Step 7: Predictions and Evaluation on Test Data
#- Hoàn thành 1 phần bài tập là kiểm tra hiệu quả của mô hình hoặc test dữ liệu
#- Chúng ta sử dụng các hàm làm dự đoán trên tập test data và RegressionEvaluate để check the RMSE value của tập dữ liệu
def transform_test(rec_model, test):
    predicted_ratings=rec_model.transform(test)
    return predicted_ratings

### 1.2.7.2. Đánh giá model
def evaluator_als(predicted_ratings):
    evaluator=RegressionEvaluator(metricName='rmse',predictionCol='prediction',labelCol='fea_rating_y')
    rmse=evaluator.evaluate(predicted_ratings)
    return rmse


#### 1.2.8. Step 8: Recommend Top item That Active User Might Like
#- Sau khi kiểm tra hiểu quả về mô hình và tuning the hyperparameters
#- Chúng ta có thể duy chuyển để xuất theo top items tới user id đó và họ có thể nhìn và thích
def recomend_top_item(data_sub, rec_model, fea_customer_id_number, fea_item_id = 'fea_item_id',fea_customer_id='fea_customer_id',how='left', lst_ = ['fea_item_id','fea_name','fea_customer_id','fea_rating_y','image','url','brand','list_price','price','rating']):
    
    unique_item=data_sub.select(lst_).distinct()

    a = unique_item.alias('a')

    fea_customer_id_number=fea_customer_id_number

    item_choose=data_sub.filter(data_sub[fea_customer_id] ==fea_customer_id_number).select(lst_).distinct()

    b=item_choose.alias('b')
    
    total_item = a.join(b, a[fea_item_id] == b[fea_item_id],how=how)

    remaining_item=total_item.where(col("b."+fea_item_id).isNull()).select(a[fea_item_id]).distinct()
   
    remaining_item=remaining_item.withColumn("fea_customer_id",lit(int(fea_customer_id_number)))
    
    recommendations=rec_model.transform(remaining_item).orderBy('prediction',ascending=False)
    
    return recommendations