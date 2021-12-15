Regression_analysis = '''

    1. Giới thiệu chung
        a. Regression analysis
            - Kỹ thuật mô hình tiên đoán (predictive modelling technique)
            - Điều tra mối quan hệ giữa dependent (target) và các independent variable (predictor)
        b. Kỹ thuật này sử dụng làm gì
            - Để dự báo (forecasting)
            - Mô hình hóa chuỗi thời gian (time series) và mối quan hệ giữa các biến
        c. Công cụ quan trọng để lập mô hình và phân tích
        => Công cụ quan trong tiên đoán dữ liệu và phân tích dữ liệu
    2. Đặc điểm và mục tiêu
        - Quy trình: Input variable => model => Output (number)
        - Mục tiêu: dự báo giá trị số (number) từ biến đầu vào input variable
        -Đặc điểm:
            + Dự đoán number từ input variable
            + Regression là supervised task
            + Target variable là giá trị số
        - Ví dụ: dự báo sản lương/ giá vàng/ giá nhà
    3. Xây dựng và áp dụng model
        - Training Phase
            * Điều chỉnh tham số model (model parameter)
            * Sử dụng dữ liệu huấn luyện (training data)
        - Testing Phase
            * Áp dụng mô hình (learned model)
            * Sử dụng dữ liệu mới (test data)
        - Đánh giá model
            * Dựa trên độ phủ của sroce của model của các Phase
            * Thời gian thực hiện
        => Mục tiêu để tránh overfitting và underfitting để xem model có phù hợp không ? => Model đó phù hợp không
    
'''
Danh_gia_mo_hinh_regression = '''
    1. Một số kỹ thuật đánh giá dùng 2 cách
        - Trực quan hóa dữ liệu: dùng distribution plot  để so sánh giữa actual value gần như khớp với nhau thì có thể dùng model
        => Ngoài ra dùng biểu đồ scatter plot 2 trùng nhau giữa actual value và predict value
        - Số liệu thống kê giữa y và y_pred có phù hợp với nhau không ?
            * Dùng thang đo thống kê: Mean Squared Error (MSE): (chêch lệch thực - dự đoán)^2/n phần tử 
            hoặc MAE (mean abssolute Error): |chêch lệch thực - dự đoán|/n phần tử
            => So sánh phương sai của 2 dự báo và thực tế để xem mức độ phương sai
            * Dùng R^2: Đo độ phù hợp của bài toán để xem mức độ phù hợp của phương sai
            1-(thực tế - dự đoán)^2/(thực tế - trung bình thực tế)^2 => MSE chia độ phân tán dữ liệu để ra đc mực độ phù hợp của model
        => Đánh giá chỉ đánh giá trên giá trị thực và giá trị dự báo
        => Phải viết ra những hàm đễ đánh giá theo thông kê và theo trực quan hóa dữ liệu
        => Phải đánh giá trên bộ dữ liệu train, bộ dữ liệu test và có thể làm ngẫu nhiên
        => Nếu sai biệt R^2 giữa train và test sai biệt tương đối với nhau, nếu sai biệt quá nhiều thì phải xem lại sự ổn định của mô hình
        => Khi cho test mô hình bằng nhau thì sẽ có tính ổn định hơn
    2. Cách xác định cần làm trong mỗi bộ dữ liệu
        - Training data: điều chỉnh các model parameter
        - Validation data: ngừng training tránh trường hợp overfiting và ước tính hiệu suất
        - Test data: Đánh giá hiệu suất trên dữ liệu mới
    3. So sánh mô hình với mô hình baseline model
        - So sánh cơ sở dụ báo với phần chêch lệch nếu nó về bằng cơ sở

'''

Linear_regression = '''
    1. Giới thiệu
        - Thuật toán đơn giản nhất trong nhóm thuật toán supervised learning
        - Hệ thống quản lý đánh giá tính dụng
        - Ý tưởng theo Regression analysis
        - Mối quan hệ được mô hình hóa tuyến tính
        => Nếu giữa 2 đại lượng biến có tuyến tính với nhau thì mới có thể sử dụng linear regression
        => Con không có mối quan hệ thì nên bỏ ra
        - Dùng Correclation lớn hơn 0.6 trở đi thì có tương quan, 0.9 trở lên thì mới có tương quan mạnh
    2. Mục tiêu
        - Làm sao tìm ra đường thẳng với 2 biến và có 1 đường thẳng tuyến tính sao cho MSE, MAE nhỏ và R2 cao nhất
    3. Hạn chế
        - Nhạy cảm với nhiễu noise: outlier, missing value, error
        - Nên xem residplot sai biệt năm trên và dưới có phân bổ điều trên và dưới và ngẫu nhiêu nó quan hệ với nhau thể nào và có thể làm 
        mô hình tuyến tính hay không => Để biết đc R2 bé hay lớn
        - Không hợp với mô hình hóa các mối quan hệ tuyến tính
    4. Random_state: dùng để cố định cho không cho dữ liệu ngẫu nhiêu cho những lần tiếp theo
    5. Không nên dùng poly vì nó tăng thuộc tính và cũng như overfitting
    
        
'''
Lua_chon_thuoc_tinh = '''
    1. Dùng thư viện K-best để chọn thuộc tính nào phù hợp với bài toán
    2. Đánh giá được feature nào là quan trọng
    3. Lựa chọn biến ố trong quy đa biến
        - Khi thêm biến số vào mô hình, SSE luôn giảm, R2 luôn tăng
        - Không nên sử dụng quá nhiều biến số sẽ gây ra hiện tượng
            * Hiện tượng quá khớp (overfitting): Cao ở bộ train và thấp ở bộ test
            * Xảy ra sử dụng nhiều biến số mà bộ mẫu quá ít
        - Hai phương pháp để khác phục overfitting
            * Sử dụng R2 hiệu chỉnh: Giá trị R2 hiệu chỉnh cho số biến độc lập
            * Tách mẫu quan sát thành 2 phần: bộ mẫu xây dựng và bộ kiểm định
'''

Mot_so_luu_y = '''
    - Các giá trị ngoại lệ (outliers): Các quan sát có giá trị khác biệt lớn so với các quan sát khác
    - Hiện tượng đa cộng tuyến: các biến độc lập (input) có tương quan lớn với nhau có thể làm sai lệch mô hình
     
'''
####################################################################################################################################
# A. BASIC ABOUT REGRESSION
####################################################################################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier # là thuật toán nhiều parama nhất
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score # Đo độ chính xác

from sklearn import model_selection
from sklearn.model_selection import KFold
# Dùng Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb 
from sklearn import svm
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import ExtraTreesRegressor
### 1.0. Trực quan hóa dữ liệu giữa y thực tế và y dự đoán
'''
    parameter:
        - y: giá trị thực tế
        - y_pred: giá trị dự đoán
    return:
        - Trả về 1 visualization
'''
def Visualize_model_reg(y, y_pred):
    result = (
        plt.figure(figsize=(12,6)),
        plt.subplot(1,2,1),
        plt.scatter(y_pred, y),
        plt.xlabel('Model Prediction'),
        plt.ylabel('True Vale'),
        plt.plot([0, np.max(y) + 2*np.min(y)], [0, np.max(y) + 2*np.min(y)], '-', color = 'r'),
        plt.subplot(1,2,2),
        sns.distplot(y, hist=False, color='r', label='True Value'),
        sns.distplot(y_pred, hist=False, color='b', label='Model Prediction', axlabel='Distribution'),
        plt.show()
    )
    return result



# 1.1. Trực quan hóa dữ liệu giữa tập training và test
'''
    parameter:
        - y_train, y_test, y_train_hat, y_test_hat: giá trị thực tế và giá trị dự đoán
    return 
        - Trực quan hóa dữ liệu
'''
def Visualize_model_reg2(y_train, y_test, y_train_hat, y_test_hat):
    ax1 = sns.distplot(y_train, hist=False, color="b", label='Train Actual')
    ax2 = sns.distplot(y_test, hist=False, color="b", label='Test Actual')
    result = (
        plt.figure(figsize=(10,5)),
        plt.subplot(1, 2, 1),

        sns.distplot(y_train_hat, hist=False, color="r", label='Train Predict', ax=ax1),
        plt.subplot(1,2,2),

        sns.distplot(y_test_hat, hist=False, color="r", label='Test Predict', ax=ax2),
        plt.show()
    )
    return result
    

# 1.2. Tính toán so sánh kết quả dự báo với dữ liệu thật
'''
    parameter:
        - y, y_pred: giá trị thực tế với giá trị dự báo
    return:
        - Trả về r2, mse, mae
'''
def Static_score_model_reg(y, y_pred):
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return r2, mse, mae   


# 1.3. lưu model và load model
'''
    - Ap dung duoc voi ca model dung trong feature engineering va model du doan cuoi cung
    parameter:
        - obj: đối tượng
        - filename: là kết quả
    return
        - saving obj
'''
def Save_Object(obj, filename):
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

# 1.4. load model
'''
    parameter: filename 
    return: Trả về đối tương mong muốn
'''
def Load_Object(filename):
    import pickle
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


'''
    1. Hàm thực hiện cross valiation
    2. Hàm thực hiện K-Folds
    3. Hàm GridSearchCV
    4. Hàm RandomSearch
    5. Hàm Select Model
'''

# 4.3.1 Hàm thực hiện cross valiation
'''
1. Vì sao cross-vadidation hữu ích ?
- Giúp đánh giá chất lượng của mô hình
- Giúp chọn mô hình hoạt động tốt nhất trên dữ liệu unsee data
- Giúp tránh overfitting and underfitting
2. Chiến lược validation
- Thông thường các chiến lược validation khác nhau tồn tại dựa trên số lượng phân tách được thực hiện trong dataset
- Train/Test split: chia thành 2 nhóm 70-30 và 80-20
- k-folds: Thuật toán chạy ổn định qua nhiều lần không
'''
'''
    paramter:
        test_size_lst = [0.3, 0.25, 0.2]
        Cross_Valiation(test_size_lst,X_Original,Y_Original)
        X_Original: dataframe X
        Y_Original: dataframe Y
    return 
        dataframe cv_df => Trả về kết quả
'''
# Xem lại lỗi hàm đã thực hiện thế nào đang có vấn đề
def Cross_Valiation(test_size_lst, X, y):
  entries = []
  models = [
    LinearRegression(),
    KNeighborsRegressor(),
    svm.SVR(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    xgb.XGBRegressor(verbosity=0),
    lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
    #MLPRegressor()
  ]
  from datetime import datetime
  from datetime import timedelta
  j = 0
  for model in models:
    scores_train = []
    scores_test = []
    times = []
    abs_scores = []
    test_sizes = []
    for i in test_size_lst:
          X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=i)
          t1 = datetime.now()
          model_name = model.__class__.__name__
          clf = model
          clf.fit(X_train,Y_train)
          t2 = datetime.now()
          d = round((t2-t1).microseconds/1000,1) #-> miliseconds
          score_train = clf.score(X_train,Y_train)
          score_test = clf.score(X_test,Y_test)
          abs_score = abs(score_train-score_test)

          scores_train.append(score_train)
          scores_test.append(score_test)
          abs_scores.append(abs_score)
          test_sizes.append(i)
          times.append(d)
            
          #print(model.__class__.__name__,scores_test, test_sizes)
          scores_train_arr = np.array(scores_train)
          j = j+1
          entries.append([model_name,
                         np.array(scores_train).mean(),
                         np.array(scores_test).mean(),
                         np.array(abs_scores).mean(),
                         np.array(times).mean(), 
                         np.array(test_sizes).mean()])
          
  cv_df = pd.DataFrame(entries,
                    columns=['model_name','score_train_mean','score_test_mean','abs|score|','time_mean','test_sizes'])
  return cv_df


# 4.3.2 Hàm thực hiện K-Folds
'''
# Các mô hình thuật toán
# Tính độ chính xác model theo:
# Logistic, Naive bayes, SVM, RandomForestClassifier, DecisionTreeClassifier
# Khi dùng KNN thì cần chọn k phù hợp trước
# Có thể viết function cho hàng này để cho nó chay
K_Folds(X=X_Original, y=Y_Original,CV=5)
'''
'''
    parameter:
        - X: dataframe input
        - Y: dataframe output
'''
# Các mô hình thuật toán
# Tính độ chính xác model theo:
# Logistic, Naive bayes, SVM, RandomForestClassifier, DecisionTreeClassifier
# Khi dùng KNN thì cần chọn k phù hợp trước
# Có thể viết function cho hàng này để cho nó chay
def K_Folds(X, y, CV=5):
  models = [
        LinearRegression(),
        KNeighborsRegressor(),
        svm.SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        xgb.XGBRegressor(verbosity=0),
        lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
        #MLPRegressor()
  ]
  cv = CV
  entries = []
  from datetime import datetime
  from datetime import timedelta
  
  for model in models:
    kq_kfold_ = []
    times = []
    clf_K = model
    t1 = datetime.now()
    model_name = model.__class__.__name__
    kfold = KFold(n_splits=cv) # Chia so phần dữ liệu train và test
    kq_kfold = model_selection.cross_val_score(clf_K,X,y,cv=kfold) # Chia bo du lieu ra 10 phần và một phần 15 mẫu để huấn luyện model
    
    t2 = datetime.now()
    d = round((t2-t1).microseconds/1000,1) #-> miliseconds

    kq_kfold_.append(kq_kfold)
    times.append(d)

    #print(model.__class__.__name__,kq_kfold_,times)
    entries.append([model_name,
                    np.array(kq_kfold_).mean(),
                    np.array(times).mean()])

  # Xem lại dữ liệu
  cv_df = pd.DataFrame(entries,
                      columns=['model_name','kq_kfold_','time_mean'])
  return cv_df


'''
1. Tunning parameter
- Đối với model đang sử dụng, điều không thể thiếu là các parameter và tất nhiên là tùy thuộc mỗi bài toán cụ thể 
số dữ liệu training đang có sẽ có cá parameter thích hợp
- Việc thử nhiều parameter khác nhau là việc rất cần thiết
=> Việc thay đổi parameter sẽ ảnh hưởng đến độ chính xác của model => Công việc của chúng ta sẽ tìm cho được các 
parameter ổn nhất và tốt nhất. Đó chính là Tunnung HyperParameter
2. Grid search
- Grid search là tập hợp các mô hình khác nhau với các giá trị tham số, năm trên 1 lưới => Đào tạo từng mô hình 
và đánh giá nó bằng cách sử dụng xác thực chéo => Chọn mô hình thực hiện tốt nhất
3. Random search
- Đúng như tên gọi, từ những parameter ta thiết lập => Sẽ chọn ngẫu nhiên các cặp parameter để tiến hành độ chính xác của model
'''


# 4.3.3 Hàm Grid_SearchC
'''
# param_grids = [param_LogisticRegression, param_GaussianNB, param_SVC, param_randomForestClassifier, 
param_DecisionTreeClassifier, param_KNeighborsClassifier]
param_grids = [param_randomForestClassifier]
    parameter:
        - Dùng để thực hiện chọn parameter cho dữ liệu các tham số hợp lý
    return 
        - Về bộ dữ liệu phù hợp
'''

def Grid_SearchCV(param_grids, X, y, CV,test_size=0.3,random_state=42):
      
  models = [
       LinearRegression(),
        KNeighborsRegressor(),
        svm.SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        xgb.XGBRegressor(verbosity=0),
        lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
        #MLPRegressor()
  ]
  cv_ =CV
  entries = []
  from datetime import datetime
  from datetime import timedelta
  for model in models:
    times = []
    best_params = []
    for param_grid in param_grids:
      X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
      start_time = datetime.now()
      CV_rfc = GridSearchCV(estimator=model,
                        param_grid=param_grid,cv=cv_)
      model_name = model.__class__.__name__
      CV_rfc.fit(X_train,Y_train)
      #print("1___",CV_rfc.fit(X_train,Y_train))
      end_time = datetime.now()
      d = round((end_time-start_time).microseconds/1000,1) #-> miliseconds
      #print("2___",d)
      best_param= CV_rfc.best_params_
      #print("3___",best_param)
      best_params.append(best_param)
      # print("4___",np.array(best_params))
      times.append(d)
      # print("5___",np.array(times).mean())

      print("Tên mô hình", model.__class__.__name__,CV_rfc.best_params_,d, "\t")

      entries.append([model_name,
                    np.array(best_params),
                   np.array(times).mean()])
    cv_df = pd.DataFrame(entries,
                     columns=['model_name','best_params','time_mean'])
    return cv_df

'''
param_randomForestClassifier = {
    'n_estimators':[30,50,100,150,200],
    'max_features':['auto','log2'],#auto = sqrt
    'criterion':['gini','entropy']}
# param_dists = [param_LogisticRegression, param_GaussianNB, param_SVC, param_randomForestClassifier, param_DecisionTreeClassifier, param_KNeighborsClassifier]
param_dists = [param_randomForestClassifier]
param_dists[0]
Random_Search(param_dists= param_dists, X=X_Original, y=Y_Original,CV=2,random_state=5,test_size=0.3, random_state_train=42)
'''
# 4.3.4 Hàm Random search
def Random_Search(param_dists, X, y, CV=5,random_state=2,test_size=0.3,random_state_train=42):
      
  models = [
    LinearRegression(),
    KNeighborsRegressor(),
    svm.SVR(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    xgb.XGBRegressor(verbosity=0),
    lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
    #MLPRegressor()
    
  ]
  
  cv_ =CV
  entries = []
  from datetime import datetime
  from datetime import timedelta
  for model in models:
    times = []
    forest_random_bests = []
    # print("1____", model)
    for param_dist in param_dists:
      X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=test_size, random_state=random_state_train)
      # print("0____", X_train['a15'])
      # print("0____", Y_train.head())
      # Thời gian bắt đầu
      start_time = datetime.now()
      # print("2____",param_dist)
      forest_random = RandomizedSearchCV(estimator=model,param_distributions=param_dist,cv=cv_,random_state=random_state)
      # print("3____",forest_random)
      model_name = model.__class__.__name__
      # print("4____",model_name)
      forest_random.fit(X_train,Y_train)
      print("5____",forest_random.fit(X_train, Y_train))
      end_time = datetime.now()
      # print("6____",end_time)
      d = round((end_time-start_time).microseconds/1000,1) #-> miliseconds
      # print("7____",d)
      forest_random_best = forest_random.best_estimator_
      # print("8____",forest_random_best)
      forest_random_bests.append(forest_random_best)
      times.append(d)

      print(model.__class__.__name__,forest_random.best_estimator_, d)

      entries.append([model_name,
                    np.array(forest_random_bests),
                    np.array(times).mean()])
    cv_df = pd.DataFrame(entries,
                       columns=['model_name','forest_random_bests','time_mean'])
    return cv_df


# 4.3.4 Select model
'''
Select_model(X=X_Original, y=Y_Original, CV=5, test_size=0.25)
parameter:
    - X: dataframe X-input
    - y: dataframe y-input
return 
    - trả về bảng cv_df kết quả
'''
def Select_model( X, y, CV=5, test_size=0.25):  
  # Tính độ chính xác model theo:
  # Logistic, Naive bayes, SVM, RandomForestClassifier, DecisionTreeClassifier
  # Khi dùng KNN thì cần chọn k phù hợp trước
  # Có thể viết function cho hàng này để cho nó chay
  models = [
    LinearRegression(),
    KNeighborsRegressor(),
    svm.SVR(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    xgb.XGBRegressor(verbosity=0),
    lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
    #MLPRegressor()
  ]
  # Số lần lặp kiểm chứng
  CV = CV
  entries = []
  # i = 0
  from datetime import datetime
  for model in models:
      scores_train = []
      scores_test = []
      times = []
      abs_scores = []
      mean_squared_error_train = []
      mean_squared_error_test = []
      for j in range(CV):
          X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=test_size)
          t1 = datetime.now()
          model_name = model.__class__.__name__
          model.fit(X_train,Y_train)
          t2 = datetime.now()
          d = round((t2-t1).microseconds/1000,1) #-> miliseconds
          score_train = model.score(X_train,Y_train)
          score_test = model.score(X_test,Y_test)
          abs_score = abs(score_train-score_test)
          
          scores_train.append(score_train)
          scores_test.append(score_test)
          abs_scores.append(abs_score)
          times.append(d)
          mean_squared_error_train_ = mean_squared_error(y_true=Y_train, y_pred=model.predict(X_train), squared=False)
          mean_squared_error_test_ = mean_squared_error(y_true=Y_test, y_pred=model.predict(X_test), squared=False)
          mean_squared_error_train.append(mean_squared_error_train_)
          mean_squared_error_test.append(mean_squared_error_test_)
      #print(model.__class__.__name__,scores_test)
          entries.append([model_name,np.array(scores_train).mean(),
                    np.array(scores_test).mean(),np.array(abs_scores).mean(), np.array(times).mean(), np.array(mean_squared_error_train).mean(), np.array(mean_squared_error_test).mean()])
      # i += 1
  cv_df = pd.DataFrame(entries,
                      columns=['model_name','score_train_mean','score_test_mean','abs|score|','time_mean', 'mean_squared_error_train', 'mean_squared_error_test'])
  return cv_df

# Chọn k trong thuật toán KNN
'''
# B. Hàm trong KNN
1. Biểu đồ Vị trí phù hợp với k
2. Biểu đồ Vị trị phù hợp với k dữ liệu train và test
3. Biểu đồ Vị trí k tuyện đối giữa k thấp nhất là k và f
4. Chọn k
5. Hàm chạy thuật toán KNN
'''
def KNN_Choose_Review(X, y, k, test_size_=0.25):
      # Kết quả
  entries = []
  # Import library



  for z in k:

    scores_train = []
    scores_test = []
    times = []
    abs_scores = []
    confusion_matrixs = []
    f1s = []
    precisions = []
    recalls = []
    classification_reports = []


    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=test_size_)

    t1 = datetime.now()

    model = KNeighborsClassifier(n_neighbors=int(z))
    model.fit(X_train,Y_train)

    t2 = datetime.now()
    d = round((t2-t1).microseconds/1000,1) #-> miliseconds


    score_train = round(model.score(X_train,Y_train),2)
    score_test = round(model.score(X_test,Y_test),2)
    abs_score = round(abs(score_train-score_test),2)
    yPred = model.predict(X_test)

    # Mức độ cân bằng dữ liệu
    confusion_matrix_ = confusion_matrix(Y_test, yPred)
    f1 = f1_score(Y_test, yPred, average='micro') # nếu target có từ 3 loại trở lên thì phải có thêm tham số averahe
    precision = precision_score(Y_test, yPred, average='micro')
    recall = recall_score(Y_test, yPred, average='micro')
    classification_report_ = classification_report(Y_test, yPred)


    scores_train.append(score_train)
    scores_test.append(score_test)
    abs_scores.append(abs_score)
    times.append(d)
    confusion_matrixs.append(confusion_matrix_)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    classification_reports.append(classification_report_)

    print("classification_report: \t",classification_report_, "\t")
    print("recall: \t",recall, "\t")
    print("precision \t",precision, "\t")
    print("f1: \t",f1, "\t")
    print("confusion_matrix: \t",confusion_matrix_, "\t")
    print("time: \t",d, "\t")
    print("abs_score: \t",abs_score, "\t")
    print("scores_test: \t",score_test, "\t")
    print("scores_train: \t",score_train, "\t")


    entries.append([np.array(scores_train),
                  np.array(scores_test),
                  np.array(abs_scores),
                  np.array(times), 
                  np.array(confusion_matrixs), 
                  np.array(f1s), 
                  np.array(precisions),
                  np.array(recalls),
                  np.array(classification_reports)])
  # Chỉ số của báo cáo, Xem mức độ cân bằng dữ liệu overfitting hay underfitting không
  indexs = pd.DataFrame(entries, columns=['score_train',
                                         'score_test',
                                         'abs|score|',
                                         'time', 
                                         'confusion_matrixs',
                                         'f1s',
                                         'precisions',
                                         'recalls',
                                        'classification_reports'])
  
  # kết quả dự bào
  result = pd.DataFrame({
    'Actual': pd.DataFrame(Y_test.values)[0].values,
    'Prediction':  pd.DataFrame(yPred)[0].values
  })
  results = [indexs, result]
  
  return results


'''
# 8. PCA - Giảm chiều dữ liệu
A. Ý tưởng thuật toán
1. Kích thước dữ liệu là 1 thách thức với phần mền máy tính => Nút cổ chai cho hiệu suất dữ liệu thuật toán ML
2. PCA phát hiện mỗi tương quan của các biến. Nếu có môi tương quan chặt chẽ giữa các biến tồn tại, nỗ lực giảm kích thước mới có ý nghĩa
3. PCA thực hiện tìm các hướng của phương sai tối đa trong dữ liệu high dimensional và chuyển vào một không gian con có chiều nhỏ hơn và giữ lại hầu hết các thông tin
4. PCA được thực hiện trong một ma trận đối xứng vuông
5. Mục tiêu là PCA làm giảm không gian thuộc tính từ một số lượng lớn các biến đến một số lượng nhỏ hơn các yếu tốt là một "Non dependent" procedure, thủ tục không phụ thuộc nó
B. Ưu điểm
1. Giảm kích thước, tăng tốc độ
2. Trực quan hóa dữ liệu
C. Nhược điểm
1. PCA không thể chỉnh hằng số (scale invariant)
2. Các hướng có phương sai lớn nhất được giả định là quan trọng nhất
3. Chỉ xem xét các phép biến đổi trực giao (các phép quay) của các biến gốc
4. PCA chỉ dựa trên vecto TB và ma trận hiệp phương sai
5. Nếu có các biến tương quan, PCA có thể giảm kính thước. Nếu không PCA sẽ không giảm kích thước
'''
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## 

def PCA_KNN_Choose_Review(X, y, k, test_size_=0.25):
      # Kết quả
  entries = []
  # Import library

  for z in k:

    scores_train = []
    scores_test = []
    times = []
    abs_scores = []
    confusion_matrixs = []
    f1s = []
    precisions = []
    recalls = []
    classification_reports = []


    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=test_size_)

    # Make an instance of the model
    pca = PCA(.95) # Giữ lại 95% thông tin
    pca.fit(X_train)
    pca.n_components_
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    t1 = datetime.now()

    model = KNeighborsClassifier(n_neighbors=int(z))
    model.fit(X_train_pca,Y_train)

    t2 = datetime.now()
    d = round((t2-t1).microseconds/1000,1) #-> miliseconds


    score_train = round(model.score(X_train_pca,Y_train),2)
    score_test = round(model.score(X_test_pca,Y_test),2)
    abs_score = round(abs(score_train-score_test),2)
    yPred = model.predict(X_test_pca)

    # Mức độ cân bằng dữ liệu
    confusion_matrix_ = confusion_matrix(Y_test, yPred)
    f1 = f1_score(Y_test, yPred, average='micro') # nếu target có từ 3 loại trở lên thì phải có thêm tham số averahe
    precision = precision_score(Y_test, yPred, average='micro')
    recall = recall_score(Y_test, yPred, average='micro')
    classification_report_ = classification_report(Y_test, yPred)


    scores_train.append(score_train)
    scores_test.append(score_test)
    abs_scores.append(abs_score)
    times.append(d)
    confusion_matrixs.append(confusion_matrix_)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    classification_reports.append(classification_report_)

    print("classification_report_pca: \t",classification_report_, "\t")
    print("recall_pca: \t",recall, "\t")
    print("precision_pca \t",precision, "\t")
    print("f1_pca: \t",f1, "\t")
    print("confusion_matrix_pca: \t",confusion_matrix_, "\t")
    print("time_pca: \t",d, "\t")
    print("abs_score_pca: \t",abs_score, "\t")
    print("scores_test_pca: \t",score_test, "\t")
    print("scores_train_pca: \t",score_train, "\t")


    entries.append([np.array(scores_train),
                  np.array(scores_test),
                  np.array(abs_scores),
                  np.array(times), 
                  np.array(confusion_matrixs), 
                  np.array(f1s), 
                  np.array(precisions),
                  np.array(recalls),
                  np.array(classification_reports)])
  # Chỉ số của báo cáo, Xem mức độ cân bằng dữ liệu overfitting hay underfitting không
  indexs = pd.DataFrame(entries, columns=['score_train',
                                         'score_test',
                                         'abs|score|',
                                         'time', 
                                         'confusion_matrixs',
                                         'f1s',
                                         'precisions',
                                         'recalls',
                                        'classification_reports'])
  
  # kết quả dự bào
  result = pd.DataFrame({
    'Actual': pd.DataFrame(Y_test.values)[0].values,
    'Prediction':  pd.DataFrame(yPred)[0].values
  })
  entries = [indexs, result]
  
  return entries

# Tạo hàm Cross validation:
def Average_RMSE_Model(model, X, y, size=0.3, cv=10):
    import time
    from sklearn.model_selection import train_test_split    
    train_rmse=[]
    train_score=[]
    test_rmse=[]
    test_score=[]
    
    duration=[]
    for n in range(1, cv+1):                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)                                             
        
        start=time.time()        
        model.fit(X_train, y_train)
        end=time.time()
        
        train_rmse.append(mean_squared_error(y_true=y_train, y_pred=model.predict(X_train), squared=False))
        test_rmse.append(mean_squared_error(y_true=y_test, y_pred=model.predict(X_test), squared=False))
        train_score.append(model.score(X_train, y_train))
        test_score.append(model.score(X_test, y_test))

        duration.append((end-start)*1000)
        
    return np.mean(train_rmse), np.mean(train_score), np.mean(test_rmse), np.mean(test_score), np.mean(duration) 

def select_model(X_train_scale, y_train):
    # Cross validation trên các model tổng quát
    lst_data = []
    train_rmses = []
    train_scores = []
    test_rmses = []
    test_scores = []
    names = []
    times = []

    models = [
            LinearRegression(),
            KNeighborsRegressor(),
            svm.SVR(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            xgb.XGBRegressor(verbosity=0),
            lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
            #MLPRegressor()        
            ]

    # Với X_train_scale
    for model in models:
        
        train_rmse, train_score, test_rmse, test_score, t = Average_RMSE_Model(model, X_train_scale, y_train, size=0.3, cv=10)
        
        lst_data.append('X_train_scale')
        train_rmses.append(train_rmse)
        train_scores.append(train_score)
        test_rmses.append(test_rmse)
        test_scores.append(test_score)
        names.append(model.__class__.__name__)
        times.append(t)

    # Với X_train
    for model in models[-2:]:
        train_rmse, train_score, test_rmse, test_score, t = Average_RMSE_Model(model, X_train_scale, y_train, size=0.3, cv=10)
        
        lst_data.append('X_train')
        train_rmses.append(train_rmse)
        train_scores.append(train_score)
        test_rmses.append(test_rmse)
        test_scores.append(test_score)
        names.append(model.__class__.__name__)
        times.append(t)
        
    # So sánh kết quả sau khi cross validation các models tổng quát
    compare_df = pd.DataFrame({'Dataset': lst_data,
                                'Model_name': names,
                                'Training RMSE': train_rmses,
                                'Testing RMSE': test_rmses,
                                'Training score': train_scores,
                                'Testing score': test_scores,                           
                                'Time':times})
    return compare_df



def check_model(df, y, choose = 'cross',CV =5,test_size=0.3 , test_size_lst = [0.2, 0.25, 0.3]):
    X_Original_organic = df.drop([y], axis=1)
    Y_Original_organic = df[y]
    # Thực hiện nhiều lần cho công việc này
    # Thêm 1 vòng lặp chạy 10 lần cho đoạn code dưới đây:
    # Dùng dictionary để lưu trữ kết quả
    # for i in range(1,11):
    # Dùng để thực hiện vòng lặp xem lại score mean trung bình để quyết định
    # 70%, 75%, 80% training and 30%, 25%, 20% test
    if choose == 'cross':
        result = Cross_Valiation(test_size_lst=test_size_lst,X=X_Original_organic,y=Y_Original_organic)
    elif choose == 'folds':
        result =K_Folds(X=X_Original_organic, y=Y_Original_organic,CV=CV)
    elif choose == 'select':
        result = Select_model(X=X_Original_organic, y=Y_Original_organic, CV=CV, test_size=test_size)
        
    return result

# Thư viện
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# Chia train/test theo tỷ lệ 80:30.
def check_model_final(df, name_y_train, type1_ ="object", type2_ =['float', 'int'], test_size=0.3, model_names = ['LinearRegression', 'RandomForestRegressor', 'SVR',\
    'GradientBoostingRegressor', 'XGBRegressor', "MLPRegressor"], 
    models = [
        LinearRegression(), 
        RandomForestRegressor(), 
        MLPRegressor(),
        KNeighborsRegressor(),
        svm.SVR(),
        GradientBoostingRegressor(),
        xgb.XGBRegressor(verbosity=0),
        lgb.LGBMRegressor(num_leaves=31,learning_rate=0.05, n_estimators=20),
        ]):
    df_train, df_test = train_test_split(df, test_size=test_size)
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.pop(name_y_train)

    X_test = df_test.iloc[:,:-1]
    y_test = df_test.pop(name_y_train)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    cat_names = list(X_train.select_dtypes(type1_).columns)
    num_names = list(X_train.select_dtypes(type2_).columns)
    # Pipeline xử lý cho biến phân loại
    cat_pl= Pipeline(
        steps=[
            ('onehot', OneHotEncoder(drop="first"))
        ]
    )
    # Pipeline xử lý cho biến liên tục
    num_pl = Pipeline(
        steps=[
            ('scaler', MinMaxScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pl, num_names), # áp dụng pipeline cho biến liên tục
            ('cat', cat_pl, cat_names), # áp dụng pipeline cho biến phân loại
        ]
    )
    # Thiết lập đánh giá chéo cho mô hình
    cv = RepeatedKFold(n_splits=10,n_repeats=3)
    # list các mô hình được lựa chọn
    models = models

    all_fit_time = []
    all_test_scores = []
    all_valid_scores = []
    all_train_scores = []
    all_rmse_scores = []
    # Đánh giá toàn bộ các mô hình trên tập K-Fold đã chia
    for model in models:
        completed_pl = Pipeline(steps=[("preprocessor", preprocessor), ("Regressor", model)])

        scores = cross_validate(completed_pl, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1, return_train_score=True)

        completed_pl.fit(X_train,y_train)
        pred = completed_pl.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred)

        all_fit_time.append(scores['fit_time'])
        all_valid_scores.append(scores['test_score'])
        all_train_scores.append(scores['train_score'])
        all_test_scores.append(r2)
        all_rmse_scores.append(rmse)
    model_names = model_names
    fittime = pd.DataFrame(all_fit_time, index=model_names).transpose()
    fittime["Box"] = "Fit time"
    trainscore = pd.DataFrame(all_train_scores, index=model_names).transpose()
    trainscore["Box"] = "Train Rscore"
    validscore = pd.DataFrame(all_valid_scores, index=model_names).transpose()
    validscore["Box"] = "Valid Rscore"
    Table_score = pd.concat([fittime,trainscore,validscore])
    Table_score_melt = pd.melt(Table_score,id_vars=["Box"])
    testscore = pd.DataFrame(all_test_scores,index=model_names, columns=["Test Rscore"]).transpose()
    rmsescore = pd.DataFrame(all_rmse_scores,index=model_names, columns=["RMSE Rscore"]).transpose()
    final_table_score = pd.concat([Table_score.groupby("Box").mean(),testscore,rmsescore])
    return final_table_score
    

########################################################################################################################################
# B. MODEL - DESIGN
#######################################################################################################################################

## for data
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ppscore
## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
import imblearn
## for deep learning
from tensorflow.keras import models, layers
import minisom
## for explainer
from lime import lime_tabular
#import shap
## for geospatial
import folium
import geopy

import tensorflow as tf
from keras import backend as K


'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()
###############################################################################
#                   MODEL DESIGN & TESTING - REGRESSION                       #
###############################################################################
'''
Fits a sklearn regression model.
:parameter
    :param model: model object - model to fit (before fitting)
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param scalerY: scaler object (only for regression)
:return
    model fitted and predictions
'''
def fit_ml_regr(model, X_train, y_train, X_test, scalerY=None):  
    ## model
    model = ensemble.GradientBoostingRegressor() if model is None else model
    
    ## train/test
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted.reshape(-1,1)).reshape(-1)
    return model, predicted



'''
Tunes the hyperparameters of a sklearn regression model.
'''
def tune_regr_model(X_train, y_train, model_base=None, param_dic=None, scoring="r2", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5)):
    model_base = ensemble.GradientBoostingRegressor() if model_base is None else model_base
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750], 'max_depth':[2,3,4,5,6,7]} if param_dic is None else param_dic                        

    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model "+scoring+":", round(random_search.best_score_, 2))
        model = random_search.best_estimator_
    
    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean "+scoring+":", round(grid_search.best_score_, 2))
        model = grid_search.best_estimator_
    
    ## K fold validation
    print("")
    print("--- Kfold Validation ---")
    Kfold_base = model_selection.cross_validate(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=scoring)
    Kfold_model = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=cv, scoring=scoring)
    print(scoring, "mean - base model:", round(Kfold_base["test_score"].mean(),2), " --> best model:", round(Kfold_model["test_score"].mean()))
    
    scores = []
    cv = model_selection.KFold(n_splits=cv, shuffle=True)
    fig = plt.figure(figsize=figsize)
    i = 1
    for train, test in cv.split(X_train, y_train):
        prediction = model.fit(X_train[train], y_train[train]).predict(X_train[test])
        true = y_train[test]
        score = metrics.r2_score(true, prediction)
        scores.append(score)
        plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
        i = i+1
    plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)], linestyle='--', lw=2, color='black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('K-Fold Validation')
    plt.legend()
    plt.show()
    
    return model



'''
Fits a keras deep/artificial neural network.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param scalerY: scaler object (only for regression)
:return
    model fitted and predictions
'''
def fit_dl_regr(X_train, y_train, X_test, scalerY, model=None, batch_size=32, epochs=100):
    ## model
    if model is None:
        ### define R2 metric for Keras
        from tensorflow.keras import backend as K
        def R2(y, y_hat):
            ss_res =  K.sum(K.square(y - y_hat)) 
            ss_tot = K.sum(K.square(y - K.mean(y))) 
            return ( 1 - ss_res/(ss_tot + K.epsilon()) )

        ### build ann
        n_features = X_train.shape[1]
        n_neurons = int(round((n_features + 1)/2))
        model = models.Sequential([
            layers.Dense(input_dim=n_features, units=n_neurons, kernel_initializer='normal', activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=n_neurons, kernel_initializer='normal', activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=1, kernel_initializer='normal', activation='linear') ])
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[R2])
        print(model.summary())

    ## train
    verbose = 0 if epochs > 1 else 1
    training = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)
    
    ## test
    predicted = training.model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted)
    return training.model, predicted.reshape(-1)



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
'''
def evaluate_regr_model(y_test, predicted, figsize=(25,5)):
    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    
    ## residuals
    residuals = y_test - predicted
    #print(type(residuals))
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    #print('max_error: ',max_error)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    #print('max_idx: ', max_idx)
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    
    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
    
    ## Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()
    
    ## Plot residuals distribution
    sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax[2], label="mean = "+"{:,.0f}".format(np.mean(residuals)))
    ax[2].grid(True)
    ax[2].set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()


###########################################################################################################################################################

###########################################################################################################################################################

from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
### Importing dataset available in sklearn
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

def check_model_final_lazy(df, y, test_size=0.3, random_state=42, choose='model'):
    df_X = df.drop([y], axis=1)
    df_y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state = random_state)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    if choose == 'model':
        return models
    elif choose == 'predictions':
        return predictions


##########################################################################################################################################################

##########################################################################################################################################################
