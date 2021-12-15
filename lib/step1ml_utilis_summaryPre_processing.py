Summary_pre_processing = '''
    1. Quy trình về xử lý dữ liệu
        Step 1 Business understanding
            - Xác định được yêu cầu bài toán
            - Hiểu nghiệp vụ cần thiết tốt nhất tìm hiểu
            - Lấy ý kiến chuyên gia
        Step2 Data understanding
            - Dữ liệu đã phù hợp với nhu cầu chưa?
            - Thêm bớt dữ liệu thuộc tính thế nào ?
            - Ý nghĩa của các thuộc tính ra sao ?
        Step3 Data prepretion
            - Vận hành các thao tác để xử lý lỗi và chuẩn hóa dữ liệu
            - Xử lý missing value và outlier không hợp lệ
            - Thêm thông tin
        Step4 Modeling
            - Phối hợp dữ liệu đã chuẩn hóa và thuật toán machine learning để build model phù hợp
        Step5 Evaluation model
            - Kiểm chứng model đã đáp ứng được yêu cầu của bài toán chưa ?
            - Nếu đúng thì bước tiếp theo
            - Nếu sai thì quay lại xem xét lại tìm hiểu các bước trước
        Step 6 Deployment
            - Kết hợp dữ liệu để cho khách hàng có thêm lựa chọn khác
            - Xem dự án có thể mở rộng thêm sau này không
    2. Tại sao lại tiền xử lý dữ liệu ? Nó cực kỳ quan trọng trong làm data science
        - Tìm ra bất thường của dữ liệu để xứ lý cho phù hợp với dữ liệu bài toán
        - Các loại dữ liệu thường được xử lý
            * Dữ liệu thường không chính xác (inaccurate data)/ Dữ liệu bị thiếu (missing data) or thuộc tính thuộc dữ liệu tổng hợp không mang nhiều ý nghĩa
            * Dữ liệu bị nhiễu (noisy data) dữ liệu bị sai ngoai lệ (outlier)
            * Dữ liệu không nhất quán (Inconsistent data) dữ liệu bị trùng lặp, do người nhập liệu, vi phạm các ràng buộc về dữ liệu
            * Các phương pháp thu thập dữ liệu thường được kiểm soát lỏng lẻo, dẫn đến các giá trị ngoài phạm vi outlier
    3. Mức độ ảnh hưởng không xử lý dữ liệu
        - Kết quả dự báo sai lệch hoàn toàn => Không dùng được model hoặc không thể làm dự báo
    4. Quy trình về pre-processing
        4.1 Quy trình chung
            - Step1: Dữ liểu thô tiềm ẩn các lỗi bên trong
            - Step2: Apply pre-processing vận dụng các chức năng nhiệm vụ để xử lý các lỗi tiềm ẩn bên trong
            - Step3: Prepared-data dữ liệu đã được xử lý
            - Step 4: Xử lý lại vòng lặp nếu chưa xử lý được hết các lỗi tiềm ẩn bên trong
        4.2 Các bước thực hiện trong pre-processing
            Step 1-Import thư viện
                - Khi nào cần dùng thì import vào
            Step 2-Đọc dữ liệu và lựa chọn các thuộc tính cần thiết
               - Hàm đọc các loại file khác nhau cần làm 1 thư viện
               - Sau khi đọc xong thì phải là 1 dataframe
               - Thuộc tính đọc vào (input) nên lựa chọn phù hợp => Nên lựa chọn các thuộc tính không liên quan đến bài toán bỏ ra ngoài
            Step3-Kiểm tra dữ liệu thiếu (Missing value)
               - Không chỉ giá trị null và những giá trị về dữ liệu thiếu
               - Ảnh hưởng rất lớn về dự đoán bài toán theo yêu cầu
            Step4-Kiểm tra dữ liệu phân loại
                - Phải chuyển các biến phân loại qua số vì các thuật toán đều làm việc với dữ liệu số
                - Một số các chuyển đổi: Label endoder, Binary Encoder, one hot code endoder
            Step5-Chuẩn hóa dữ liệu (Data standardizing)
                - Tập dữ liệu đầu thường chỉ chứa các thuộc tính số và do đó có thể cần phải chia tỷ lệ (Feature scaling) cho các thuộc tính 
                trong dữ liệu trước khi thực hiện công việc tiếp theo như PCA, Kmeans...
                - Chia tỉ lệ theo phương pháp giới hạn nhập vi của các thuộc tính để chung có thể được so sánh dựa trên các căn cứ chung. 
                Một số cách để chia tỉ lệ là Standard Scaler, Min - max scaler...
            Step 6-Chuyển đổi dữ liệu
                - PCA tranformation giảm kích thước chiều không gian tính năng trong khi vẫn giữ lại các thông tin cần thiết
                - Chuyển đổi dữ liệu sao cho mức độ tương quan biến input và output phụ thuộc với nhau
            Step 7-Chia dữ liệu
                - Chia dữ liệu Training và testing
                - Xem mức độ cần bằng dữ liệu nếu biến output là biến categorical
'''
####################################################################################################################################################################
#                                                                      SUMMARY PRE-PROCESSING                                                                      #
####################################################################################################################################################################
from inspect import Parameter
import pandas as pd 
import warnings
import pandas_profiling as pp # tổng quan ban đầu về dữ liệu => Cài trên này
warnings.filterwarnings('ignore')
import re
import string
import langid # Dùng để xem ngôn ngữ nước nào
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import time

Summary_function_pre_processiong = '''
    A. Data exploration - Khám phá dữ liệu
    1. Khám phá dữ liệu vì sao thực sự quan trọng ?
        - Một bước trong quá trình khám phá dữ liệu (Data clearning)
        - Khi làm việc với machine learning phải vật lộn để cải thiện độ chính xác của mô hình
        - Khám phá dữ liệu là bước cực kỳ quan trọng => Ảnh hưởng đến chất lượng input (biến đầu vào) => Dự báo được biến đầu ra
        - Chất lượng input đầu vào sẽ quyết định chất lượng biến đầu ra target (input)
        - Thời gian khám phá dữ liệu và làm sách dữ liệu chiếm phần lớn dự án đến gần 70% đến 80%       
'''
Missing_value = '''
    3. Các xử lý dữ liệu thiếu (Missing value)
        3.1. Tại sao phải xử lý dữ liệu thiếu ?
            - Tập training data set là tập không thể thiếu dữ liệu vì nó sẽ ảnh hưởng đến sức mạnh của model
        3.2. Một số nguyên nhân dữ liệu bị thiếu ?
            - Khai thác dữ liệu (data extraction): có thể ảnh hưởng đến quá trình truy xuất
            - Thu thập dữ liệu (Data colection): Thu thập trong quá trình sửa chửa giai đoạn này khó để sửa hơn
        3.3. Cách xử lý dữ liệu thiếu
            - Xóa (deletion): giảm kích thước mẫu, nên cần xem lại
            - Dùng Mean/mode, median
            - Dùng mô hình dự báo KNN hoặc 1 mô hình ML khác
            - Tự build 1 model để dự đoán dữ liệu
        3.4. Có 2 nhược điểm cho phương pháp dự báo
            - Các giá trị ước tích của mô hình thường xử lý tốt hơn các giá trị thực
            - Nếu không có mối quan hệ giữa các thuộc tính trong tập dữ liệu và 
            thuộc tính có các giá trị bị thiếu thì mô hình sẽ không chính xác khi ước tính các giá trị thiếu
            - Có thể sử dụng interpolate() để nội suy tuyến tính cho các giá trị bị thiếu
        3.5. Ưu điểm và nhược điểm của thuật toán KNN
            - Ưu điểm:
                * KNN có thể dự đoán cả hai thuộc tính định tính và định tính
                * Không cần tạo mô hình dự báo cho từng thuộc tính có dữ liệu thiếu
                * Có thể xử lý dễ dàng dữ liệu thiếu
                * Cấu trúc tương quan của dữ liệu được xem xét
            - Nhược điểm
                * KNN tốn thời gian trong việc phân tích dữ liệu lớn.
                Nó tìm kiếm dữ liệu thông qua tất cả dữ liệu để tìm các trường hợp tương tự nhất
                * Lựa chọn giá trị k là rất quan trọng
        => Có thể xử lý missing số và missing NLP
'''
## A. LOAD DATA
### 1.1. Load data: dùng để load data các file đuôi csv, xlsx, json
'''
    file_dir: đường dẫn file năm ở đâu
    names: tên thuộc tính mong muốn
    return: kết quả là 1 dataFrame
'''
def loadData(file_dir="", names=""):
    #try:
        file_dir = file_dir.lower()
        if file_dir.endswith("csv"):
            df = pd.read_csv(file_dir, names=names)
            return df
        elif file_dir.endswith("xlsx"):
            df = pd.read_excel(file_dir, names=names)
            return df
        elif file_dir.endswith("json"):
            df = pd.read_json(file_dir, names=names)
            return df
        else:
            print("Please see file and path")
    #except Exception as failGeneral:
    #    print("Fail system, please call developer...", type(failGeneral).__name__)
    #    print("Mô tả:", failGeneral)
    #finally:
    #    print("close")

## B. TỔNG QUAN VỀ DỮ LIỆU
### 1.2. Tổng quan về thông tin ban đầu: info; nan; head; tail; shape; null; profile; dtypes... theo if elif else
'''
    df: bảng dữ liệu
    choose: chọ các thông tin cần muốn xem [info, nan, head, tail, shape, null, profile, dtypes, columns]
    head: hiện thị 10 giá trị đầu tiên
    tail: hiện thị 10 giá trị cuối
    Riêng choose: profile cho chúng ta biết sơ qua về dữ liệu nhưng nên chỉnh sửa dữ liệu trước sau đó có nhìn lại và nhận xét về dữ liệu
    return: thông tin cần thiết
'''
def startInformation(df, choose, head =10, tail = 10):
    try:
        # Xem info
        if choose == "info":
            info = df.info()
            return info
        # Xem dữ liệu trống
        elif choose == "nan":
            nan = df.isna().sum()
            return nan
        # Xem 10 dữ liệu đầu
        elif choose == "head":
            head = df.head(head)
            return head
        # Xem 10 dữ liệu đầu
        elif choose == "tail":
            head = df.tail(tail)
            return head
        elif choose == "shape":
            shapes_ = df.shape
            return shapes_
        elif choose == "null":
            # Xem giá trị null
            null_ = df.isnull().sum()
            return null_
        elif choose == "profile":
            # Nhìm tổng quan về báo cáo
            profile = pp.ProfileReport(df)
            return profile
        elif choose == "dtypes":
            dtypes_ = df.dtypes
            return dtypes_
        elif choose == "columns":
            columns = df.columns
            return columns
        else:
            print("Please see file")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.3. % dữ liệu trùng: Xem tỉ lệ % dữ liệu trùng
'''
    df: dataframe đưa vào biến
    return: tỉ lệ chiếm % dữ liệu duplicate trong dữ liệu
'''
def percentDuplicates(df):
    try:
        shapeBefore = df.shape
        countBefore = shapeBefore[0]
        data = df.drop_duplicates()
        shapeAfter = data.shape
        countAfter = shapeAfter[0]
        variableCount = countBefore - countAfter
        if variableCount == 0:
            result_ = 0
            return result_
        elif variableCount != 0:
            result_ = round(variableCount/countBefore,5)
            return  result_
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.4. Loại bỏ dữ liệu trùng
'''
    df: dataframe trong dữ liệu
    return: Xóa hết dữ liệu duplicate in dataframe => Trả về dataframe
'''
def deleteDuplicates(df):
    try:
        data = df.drop_duplicates()
        return  data
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.5. Xử lý dữ liệu (missing values)
'''
    df: dataframe
    choose: chọn print in ra từng dòng; table: ra bảng để chung ta xem mức đô dữ liệu
    return table: nó là dataframe; con nều là print: nó sẽ in theo dạng consol
'''
def checkDtypesDataAndMissingvalues(df, chooses, index_=2, columns1 = ['Kiểm tra biến','giá trị', 'Số biến','Kiểu dữ liệu']):
    try:
        lists_ = []
        for i in df.columns:
            if len(df[i].unique()) > index_:
                if chooses == "print":
                    print('\033[4m'+'Kiểm tra biến', i +'\033[0m', ': ', len(df[i].unique()), 'giá trị,', 
                    df[i].sort_values().unique(), ', dtype:', df[i].dtypes)
                    #list_ = [i, len(df[i].unique()), df[i].sort_values().unique(), df[i].dtypes]
                    #lists_.append(list_)
                    #result_ = pd.DataFrame(list_)
                    #return lists_
                elif chooses == "table":
                    list_ = (i, len(df[i].unique()), df[i].sort_values().unique(), df[i].dtypes)
                    lists_.append(list_)
                    result_ = pd.DataFrame(lists_, columns=columns1)
                    #result_.set_index('Kiểm tra biến')
                else:
                    print("Please check again !!!!")
            else:
                if chooses == "print":
                    print('\033[4m'+'Kiểm tra biến', i +'\033[0m', ': ', len(df[i].unique()), 'giá trị,', df[i].sort_values().unique(), ', dtype:', df[i].dtypes)
                    #list_ = [i, len(df[i].unique()), df[i].sort_values().unique(), df[i].dtypes]
                    #lists_.append(list_)
                    #result_ = pd.DataFrame(list_)
                    #return lists_
                elif chooses == "table":
                    list_ = (i, len(df[i].unique()), df[i].sort_values().unique(), df[i].dtypes)
                    lists_.append(list_)
                    result_ = pd.DataFrame(lists_, columns=columns1)
                    #result_.set_index('Kiểm tra biến')
                else:
                    print("Please check again !!!!")
        return result_
            
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.6. Gọi dữ liệu categorical để xem sự hợp lý của nó
'''

'''
def output_feature_missing_value(df, lst_view, key, columns):
    lsts_ = df[lst_view][key]
    result_ = pd.DataFrame(lsts_, columns=['Số biến'])
    return result_


## D. XỬ LÝ MISSING VALUE
### 1.11. Xử lý dữ liệu số với mean/ mode/ median => thay thế dữ liệu missing value bằng các giá trị mean/median/mode
'''
    - parameter:
        df: dataframe
        choose: [mean, median, mode, zero]
        lst_continuous: list thuộc tính cần thay đổi
        d: có thể chạy vòng for nếu khi thay vị xuất hiện dữ liệu 1.999.99 mà là dữ liệu 1.999.999.99 => d hơn 3
        findString: Giá trị tìm kiếm
        dillValue_ = giá trị muồn thay
    - return: dataframe
'''
def changeMisingValueContinuous(df, choose = "Không chọn", lst_continuous=[], value_change=0, findString='.', dillValue_='', d=1):
    try:
        # Xử lý dữ liệu thiếu của các biến liên tục bằng giá trị mean_i
        lst_missing = lst_continuous
        for i in lst_missing:
            if choose == "mean":
                # https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
                mean_i = df.loc[pd.to_numeric(df[i], errors='coerce').isnull()==False,i].astype(float).mean()
                df.loc[pd.to_numeric(df[i], errors='coerce').isnull(),i] = mean_i
            elif choose == "median":
                median_i = df.loc[pd.to_numeric(df[i], errors='coerce').isnull()==False,i].astype(float).median()
                df.loc[pd.to_numeric(df[i], errors='coerce').isnull(),i] = median_i
            elif choose == "mode":
                mode_i = df.loc[pd.to_numeric(df[i], errors='coerce').isnull()==False,i].astype(float).mode()
                df.loc[pd.to_numeric(df[i], errors='coerce').isnull(),i] = mode_i
            elif choose == "zero":
                df.loc[pd.to_numeric(df[i], errors='coerce').isnull(),i] = value_change
            elif choose == "number":
                value_ = df.loc[df[i].str.count('.')>=2,i].str.replace(findString,dillValue_,d)
                df.loc[df[i].str.count('.')>=2,i] = value_
            elif choose == "other":
                df.loc[df[i].isnull(),i] = 'other'
        return df         
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.12. Filter special character: Tỉm kiếm các ký tự đặc biệt trong kiểu dữ liệu object
'''
    parameter:
        - df => Dataset
    return: về 1 dataframe nên ta có thể kết hợp được kết quả
'''
def filterSpecialCharacter(df):
    try:
        lists_ = []
        data = "Bang"
        specialCharacter = ['^','<','>','{','}','""','/','|',';',':','.',',','~','!', \
            '?','@','#','$','%','=','&','*','(',')','\\','[','¿','§','«',\
            '»','ω','⊙','¤','°','℃','℉','€','¥','£','¢','¡','®','©','0','-','9','_','+',']','*','$','--']

        for chara in specialCharacter:
            ten = data.join(chara)
            for i in df.columns:
                ten = df.loc[df[i] == chara]
                lists_.append(ten)
        result = deleteDuplicates(df = pd.concat(lists_))
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 1.14.1. Chuyển đổi kiểu dữ liệu cho thuộc tính: change astype for feature
'''
    prameter: 
        - df: là dữ liệu dataframe
        - lst_int: Là dữ liệu muốn chuyển qua kiểu dữ liệu int
        - lst_float: là dữ liệu muốn chuyển qua kiễu dữ liệu float
    return trả về dataframe mong muốn
'''
def changeToAstype(df, lst_int, lst_float):
    # Chuyển thành kiểu số int và float cho các biến kiểu số
    df[lst_int] = df[lst_int].astype(int)
    df[lst_float] = df[lst_float].astype(float)
    return df

### 1.14.1. Chuyển đổi kiểu dữ liệu cho thuộc tính: change astype for feature
'''
    prameter: 
        - df: là dữ liệu dataframe
        - lst_int: Là dữ liệu muốn chuyển qua kiểu dữ liệu int
        - lst_float: là dữ liệu muốn chuyển qua kiễu dữ liệu float
    return trả về dataframe mong muốn
'''
def change_type_lst(df, lst_change, choose = 'int'):
    if choose == 'int':
        for i in lst_change:
            df[i] = df[i].astype(int)
    elif choose == 'float':
        for i in lst_change:
            df[i] = df[i].astype(float)

### 1.14.1. Chuyển đổi kiểu dữ liệu cho thuộc tính: change astype for feature
'''
    prameter: 
        - df: là dữ liệu dataframe
        - lst_float: thuộc tính muốn chuyển qua 
    return trả về dataframe mong muốn
'''
def changeToAstype_date(df, feature_date):
    # Chuyển thành object sang qua date
    df[feature_date] = pd.to_datetime(df[feature_date])
    return df

### 1.15. Xử lý dữ liệu thiếu của các biến phân loại: add value for value failing
def misingValueCategorical(df, list_category, choose = "other"):
    #try:
        # Xử lý dữ liệu thiếu của các biến phân loại
        specialCharacter = ['^','<','>','{','}','""','/','|',';',':','.',',','~','!', \
            '?','@','#','$','%','=','&','*','(',')','\\','[','¿','§','«',\
            '»','ω','⊙','¤','°','℃','℉','€','¥','£','¢','¡','®','©','0','-','9','_','+',']','*','$','--','?']
        for chara in specialCharacter:
            for i in list_category:
                #print(i)
                if choose =="mode":
                    mode = df[i].mode().values[0]
                    #print(mode)
                    df.loc[df[i] == chara, i] = mode
                    #print('\033[4m'+'Kiểm tra biến', df[i].value_counts())
                elif choose=="other":
                    df.loc[df[i] == chara, i] = 'other'
                elif choose=="change":
                    df.loc[df[i] == chara, i] =''
        return df
    #except Exception as failGeneral:
    #    print("Fail system, please call developer...", type(failGeneral).__name__)
    #    print("Mô tả:", failGeneral)
    #finally:
    #    print("close")

### 1.16. Xem mực độ chạy dữ liệu với thời gian ban đâu
def time_project():
    time_ = time.time()
    return time_

### 1.12. Filter special character: Tỉm kiếm các ký tự đặc biệt trong kiểu dữ liệu object
'''
    parameter:
        - df => Dataset
    return: về 1 dataframe nên ta có thể kết hợp được kết quả
'''
def filterSpecialCharacter_one_feature(df, lst_feature):
    try:
        lists_ = []
        data = "Bang"
        specialCharacter = ['^','<','>','{','}','""','/','|',';',':','.',',','~','!', \
            '?','@','#','$','%','=','&','*','(',')','\\','[','¿','§','«',\
            '»','ω','⊙','¤','°','℃','℉','€','¥','£','¢','¡','®','©','0','-','9','_','+',']','*','$','--']

        for chara in specialCharacter:
            ten = data.join(chara)
            ten = df.loc[df[lst_feature] == chara]
            lists_.append(ten)
        result = deleteDuplicates(df = pd.concat(lists_))
        return result[lst_feature]
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

# 1.14. Chuyển dổi thuộc tính
'''
    Parameter:
        df: chọn dataframe
        lst_category: thuộc tính cần chuyển đổi
        names: tên côt
    Return
        Trả về 1 dataframe
'''
def change_feature_seriesToDataframe(df, lst_feature, names):
    df_length_ = df[lst_feature].to_frame(name=names)
    return df_length_

# 1.15. Đổi kiểu dữ liệu cho thuộc tính
'''
    parameter
        df - dataframe
        lst_int: kiểu định dạng là int
        lst_float: kiểu định dạng là float
        lst_object: kiểu dữ liệu object
    return
        Trả về dataframe mới
'''

def changeToAstypeAll(df, lst_int, lst_float, lst_object):
    # Chuyển thành kiểu số int và float cho các biến kiểu số
    lst_int = lst_int
    lst_float = lst_float
    df[lst_int] = df[lst_int].astype(int)
    df[lst_float] = df[lst_float].astype(float)
    df[lst_object] = df[lst_object].astype(object)
    return df

# 1.16. Thay đổi 1 số đơn vị tính: UoM_price_house và UoM_amount
'''
    parameter:
        - df: dataframe
        - id_index: cần thay đổi
        - feature: cột cần thay đổi
        - values: giá trị cần thay đổi
    return
        - Trả về dataframe
'''
def change_cell_value(df, id_index, feature, values):
    df.loc[id_index, feature] = values
    return df

# 1.17. +-*/ với các cột trong dataframe
'''
    parameter:
        - df: dataframe
        - calculation: Phép toán +-*/ ; Cộng trừ nhân chia
        - values: Giá trị cần thêm vào
        - feature_condition: Thuộc tính điều kiện chọn
        - condition_text: Điều kiện text
        - feature_old: thuộc tính củ
        - feature_new: Thuộc tính mới
    return
        - Trả về dataframe
'''

def calculation_value_new_feature(df, operation, values, feature_condition, condition_text, feature_old , feature_new):
    value = df[feature_old]
    if operation == 'chia':
        df.loc[df[feature_condition] == condition_text,feature_new] = value/values
        df.loc[df[feature_condition] != condition_text,feature_new] = value
    elif operation == 'nhân':
        df.loc[df[feature_condition] == condition_text,feature_new] = value*values
        df.loc[df[feature_condition] != condition_text,feature_new] = value
    elif operation == 'cộng':
        df.loc[df[feature_condition] == condition_text,feature_new] = value+values
        df.loc[df[feature_condition] != condition_text,feature_new] = value
    elif operation == 'trừ':
        df.loc[df[feature_condition] == condition_text,feature_new] = value-values
        df.loc[df[feature_condition] != condition_text,feature_new] = value
    else:
        print("Please check again !!!!")
    return df

# 1.18. Hàm tính + - * / chủa 2 series 
def calculation_2_feature_to_new_fearture(df, calculation, name_feature_new, name_feature_one, name_feature_two):
    
    if calculation == '+':
        df[name_feature_new] = round(df[name_feature_one] + df[name_feature_two],2)
    elif calculation == '-':
        df[name_feature_new] = round(df[name_feature_one] - df[name_feature_two],2)
    elif calculation == '*':
        df[name_feature_new] = round(df[name_feature_one] * df[name_feature_two],2)
    elif calculation == '/':
        df[name_feature_new] = round(df[name_feature_one] / df[name_feature_two],2)
    
    return df

# 1.19. Thêm 1 thuộc tính mới vào trong dữ liệu
'''
    parameter:
        - df: dataframe
        - feature_condition:feature thể hiện muốn tiềm kiếm điều kiện
        - condition_text: điều kiện text
        - feature_new: tên thuộc tính mới
        - values_true: giá trị trả về muốn trả về true
        - values_false: giá trị trả về muốn trả về false
    Return
        - Trả về dataframe
'''
def calculation_value_new_feature_categorical(df, feature_condition, condition_text, feature_new, values_true, values_false):
  
    df.loc[df[feature_condition].str.contains(condition_text) == True, feature_new] = values_true
    df.loc[df[feature_condition].str.contains(condition_text) == False, feature_new] = values_false
    
    return df


# 1.20. Thêm 1 thuộc tính mới vào trong dữ liệu
'''
    parameter:
        - df: dataframe
        - feature_condition:feature thể hiện muốn tiềm kiếm điều kiện
        - condition_text: điều kiện text
        - feature_new: tên thuộc tính mới
        - values_true: giá trị trả về muốn trả về true
        - values_false: giá trị trả về muốn trả về false
    Return
        - Trả về dataframe
'''
def calculation_value_new_feature_condition(df, feature_condition, condition_text, feature_new, feature_value_old, feature_value_new):
  
    df.loc[df[feature_condition].str.contains(condition_text) == True, feature_new] = df[feature_value_new]
    df.loc[df[feature_condition].str.contains(condition_text) == False, feature_new] = df[feature_value_old]

    return df
# 1.21. Xử lý date trong dataframe
'''
    parameter:
        - df: dataframe
        - feature_date: thuộc tính ngày
    return:
        - dataframe
'''
def date_add_feature(df, feature_date):
    df['fea_year'] = pd.DatetimeIndex(df[feature_date]).year
    df['fea_month'] = pd.DatetimeIndex(df[feature_date]).month
    df['fea_day'] = pd.DatetimeIndex(df[feature_date]).day
    df['fea_weekofyear'] = pd.DatetimeIndex(df[feature_date]).weekofyear
    df['fea_daily'] = pd.DatetimeIndex(df[feature_date]).weekday
    return df

# 1.22. Xử lý combine data feature
'''
    parameter: 
        - lst_concat: là 1 list các dataframe
    return:
        - Trả về 1 dataframe mới
'''
def dataframe_concat(lst_concat = []):
    df_new = pd.concat(lst_concat, axis=1)
    return df_new

# 1.23. Kiểm tra các thuộc tính có giá trị âm hoặc dương
'''
    parameter:
        - df: dataframe
        - lst_negative: list negative
        - choose: chọn +/-
    return:
        - dataframe
'''
def check_negative_value(df, lst_negative = [], value_compare = 0, choose = 'negative'):
    lst_result = []
    if choose == 'negative':
        for i in lst_negative:
            df[i] = df.loc[df[i] < value_compare]
    elif choose == 'positive':
        for i in lst_negative:
             df[i] = df.loc[df[i] >= value_compare].shape
    
    return df

# 1.24. Công trừ nhân chia 2 cột
'''
    parameter:
        - df: dataframe
        - name_feature1: tên cột 1
        - name_feature2: tên cột 2
        - name_new_feature: tên feature mới
        - choose: cộng trừ nhân chia
    return:
        - df mới
'''
def calculation_2_feature(df, name_feature1, name_feature2, name_new_feature,  choose = 'cộng'):
    if choose == 'cộng':
        df[name_new_feature] = df[name_feature1] + df[name_feature2]
    elif choose == 'trừ':    
        df[name_new_feature] = df[name_feature1] - df[name_feature2]
    elif choose == 'nhân':    
        df[name_new_feature] = df[name_feature1] * df[name_feature2]
    elif choose == 'chia':    
        df[name_new_feature] = df[name_feature1] / df[name_feature2]
    else:
        print("Please check again !!!!!!!!")
    return df

# 1.25. Các hàm tính gộp trong dataframe   
def groupby_mean(x):
    return x.mean()
def groupby_count(x):
    return x.count()
def purchase_duration(x):
    return (x.max() - x.min()).days
def avg_frequency(x):
    return (x.max() - x.min()).days/x.count()
groupby_mean.__name__ = 'avg'
groupby_count.__name__ = 'count'
purchase_duration.__name__ = 'purchase_duration'
avg_frequency.__name__ = 'purchase_frequency'

# 1.26. Groupby dữ liệu
def groupby_feature(df, name_groupby):
    summary_df = df.reset_index().groupby(name_groupby).agg({
    'Sales': [min, max, sum, groupby_mean, groupby_count],
    'InvoiceDate': [min, max, purchase_duration, avg_frequency]
    })
        
# 1.27. headmap: thể hiện tinh tương quan của dữ liệu
'''
    parameter:
        - df: là dataframe
        - width: chiều rộng của hình
        - height: chiều dai của hình
    return:
        - Hình ảnh cần thiết
'''
def corr_headmap(df, width =12, height=6 ):
    plt.figure(figsize=(width,height))
    return sns.heatmap(df.corr(),cmap='coolwarm',annot=True)

# 1.28. Tương quan 2 biến với biên phân loại
'''
    parameter:
        - df: dataframe
        - varable_continious_x: biến continious cột x
        - varable_continious_y: biến continious cột y
        - varable_categorical: biến categorical phân loại
    return:
        - Trả về ảnh mong muốn
'''
def corr_lmplot(df, varable_continious_x, varable_continious_y, variable_categorical):
    return sns.lmplot(data=df, x = varable_continious_x, y = varable_continious_y, hue =variable_categorical)

# 1.29. Xem xét biến động giá bình quân của 2 loại bơ qua các năm
'''
    parameter:
        - df: dataframe
        - time: month, year, ...time series
        - variable_continious: price, ....
        - variable_categorical: phân loại cụm
        - kind: loại biểu đồ hiển thị
        - size: kích cở hình
    return
        - Hình ảnh
'''
def diff_varibale_factorplot(df, time, variable_continious, variable_categorical, kind="box",size=7, aspect=1.5):
    result = sns.factorplot(time, variable_continious,data=df,
                    hue=variable_categorical,
                    kind=kind,size=size, aspect=aspect
                )
    return result

def diff_varibale_factorplot_dodge(df, time, variable_continious, variable_categorical, dodge=True,size=7, aspect=1.5):
    result = sns.factorplot(time, variable_continious,data=df,
                    hue=variable_categorical,
                    dodge=dodge,size=size, aspect=aspect
                )
    return result

def barplot_feature(df, x_time, variable_continious, variable_categorical):
    plt.figure(figsize=(20,10))
    return sns.barplot(data = df, x = x_time, y = variable_continious, hue =variable_categorical)

# 1.30. Xem xét biến động giá của 2 loại bơ qua thời gian của dữ liệu
'''
    parameter:
        - df: dataframe
        - variable_continious: tên biến liên tục
        - variable_categorical: tên biên phân loại
        - variable_type1: tên phân loại 1
        - variable_type2: tên phân loại 2
        - date_: time series
    return:
        - Trả về hình ảnh
'''
def fluctuations_variable_continiou_time(df, variable_continious,variable_categorical, variable_type1, variable_type2, date_):
    byDate_conv=df[df[variable_categorical]==variable_type1].groupby(date_).mean(),
    byDate_org=df[df[variable_categorical]==variable_type2].groupby(date_).mean(),
    result = (
        plt.figure(figsize=(20,8)),
        byDate_conv[variable_continious].plot(),
        byDate_org[variable_continious].plot(),
        plt.title(variable_continious),
    )
    return result


# 1.31. Pie chart cho tổng sản lượng tiêu thụ của 2 loại bơ qua các năm
'''
    parameter:
        - df: dataframe
        - variable_categorical: type dữ liệu hiển thị
        - new_name: tên thuộc tính mới
        - function_: sum, max, min...
        - title: hiển thị tên title
    return:
        - hiện thì hình ảnh dữ liệu theo chart pie
'''
def pie_chart_variable_continious(df, variable_categorical, new_name, function_, title):
    volume = df.groupby(variable_categorical).agg({new_name:function_})
    result = (
        plt.figure(figsize=(7,7)),
        plt.pie(volume[new_name], labels = volume.index, autopct='%1.2f%%'),
        plt.title(title),
        plt.legend(),
        plt.show()
    )
    return result
# 1.32. groupby dữ liệu
'''
    parameter:
        - df: dataframe
        - variable_categorical: biến phân loại
        - variable_continious1: biến 1 liên quan đến bài toán
        - variable_continious2:
        - variable_continious3:
        - function_: mean, max, min, sum...
        - X: hàng tử x với gì mong muốn
    return:
        - trả về 1 dataframe mới
        
'''
def group_by_data(df, variable_categorical, variable_continious1, variable_continious2, variable_continious3, function_, X=1):
    df_new = df.groupby(variable_categorical).agg({variable_continious1:function_,variable_continious2:function_,variable_continious3:function_})
    df_new[variable_continious1] = df_new[variable_continious1]*X
    df_new = df_new.sort_values(by=[variable_continious2,variable_continious3,variable_continious1],ascending=False).reset_index()
    return df_new
    

######################################################################################################################################################
#                                                              SUMMARY DATA EXPLORATION                                                              #
######################################################################################################################################################

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import warnings
import pandas_profiling as pp # tổng quan ban đầu về dữ liệu => Cài trên này
warnings.filterwarnings('ignore')
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.stats import kurtosis, skew 

Summary_data_exploration = '''
    2. Các bước thực hiện Data exploration
        2.1. Xác định thuộc tính/ biến (Variable indentification)
                - Nhiệm vụ: Loại dữ liệu; mỗi quan hệ giữa các thuộc tính; thuộc tính nào hữu ích; loại bỏ các thuộc tính không cần thiết
                - Xác định các thuộc tính: Xác định thuộc tính hữu ích kinh tế; tiếp thị; phân tích hiệu suất chính (KPIs) và xác định biến đâu ra cần xác định
                - Các bước thục hiện khi phân tích thuộc tính/Biến
                    Step1: Xác định các biến đầu vào (input) và output
                        * Cách dễ nhất để xác định biến input là xác định biến output
                    Step2: Kiểu dữ liệu của thuộc tính
                        * Kiểu dữ liệu mặc định thế nào
                        * Hiểu về thuộc tính dữ liệu và bản chất của dữ liệu
                        * Ưu và nhược điểm của mỗi thuộc tính
                    Step3: Xác định loại thuộc tính phân loại hay liên tục
                        * Thuộc tính phân loại kiểu số hay kiểu chuỗi
                        * Cần dựa vào ý nghĩa của thuộc tính
                - Kiểu dữ liệu thường có 4 loại
                    * Numerical Data: Kiểu rồi rạc hoặc liên tục có logic/ xem mức độ range = max - min
                    * Categorical Data: Loại theo thứ tự theo tự nhiên hoặc loại không theo thứ tự/ text: spam hoặc ham
                    * Time seris Data: Biến động theo thời gian và liên tục
                    * Text: Dạng văn bản NLP
        2.2. Phân tích đơn biến (Univariable analysis)
            2.2.1. Biến liên tục (Continuous variables)
                Thứ 1: Tìm hiểu xu hướng trung tâm và sự lây lan của biến
                Thứ 2: Được đo bằng các thước đo
                    - Central tendency - Xu hướng trung tâm và lây lan của biến
                        * Mean: giá trị trung bình
                        * Median: trung vị của dữ liệu
                        * Mode: giá trị xuất hiện nhiều
                        * Min: Giá trí thấp nhất
                        * Max: Giá trị cao nhất
                    - Measure of dispersion - Độ phân tán của dữ liệu
                        * Range = Max - Min: khoảng cách
                        * Quartile liên quan đến tứ phân vị: 25%, 50%, 75%
                        * IQR: Khoảng cách dữ liệu năm trong vùng median 25% và median 75% hay Q1 và Q3 cho biết các giá trị outlier hợp lý và không hợp lý
                        * Variable (đo khoảng cách dữ mỗi số): mức độ phân tán của dữ liệu
                        * Standard Deviation: độ lệch chuẩn, trung tâm lệch chuẩn, lệch trái, lệch phải...
                        * Skewness: Phân phối chuẩn hay không ? => nếu trả về 0 phân phổi chuẩn, lớn >0 lệch phải; < 0 lệch trái.
                        * Kurtosis: =0 thì phân phổi chuẩn >0 nhọn hơn phân phối chuẩn; <0 thì nó bẹt hơn phân phối chuẩn
                    - Visualiation methods
                        * Histogram: Giá trị phối theo cột
                        * Box plot: thuộc tính có giá trị outlier  hay không
                        * Distributionplot: Phân phối trung tâm bị lệch phải hay lệch trái so với trung tâm
                Thứ 3: Mục đích
                    - Làm nỗi lên giá trị thiếu hoặc missing value hoặc ngoại lệ/outlier
            2.2.2. Biến phân loại (Categorical variables)
                Thứ 1: Sử dụng tần số để hiểu phân phối cùng loại/ tỉ lệ % theo giá trị hàng mục hoặc theo đếm dữ liệu
                Thứ 2: Được đo bằng các thước đo
                    - Đếm bằng value_count theo từng category => Xem được mức độ cần bằng của dữ liệu theo tần suất  
                Thứ 3: Trực quan hóa dữ liệu
                    - Bar chart để trực quan hóa xem mức độ dữ liệu
            2.2.3. Mục đích của việc phân tích đơn biến
                - Sử dụng để làm nổi bật các giá trị bị thiếu và ngoại lệ
                 
'''

## A. Phân tích thuộc tính

### 2.1. Các thuộc tính input kiểu số /kiểu chuỗi: Xem lại kiểu dữ liệu của bài toán
'''
    dataframe:
        - df: dataframe mông muốn
        - lst_input: là list input đầu vào
    return
        - dataframe các thuộc tính
'''
def defineDtypeFeatures(df, lst_input, choose = "object"):
    try:
        if choose == "object":
            print('Thuộc tính kiểu chuỗi:', df[lst_input].select_dtypes(include=['object']).columns)
            lists_ = df[lst_input].select_dtypes(include=['object']).columns
            result = pd.DataFrame(lists_, columns=['Thuộc tính kiểu chuỗi'])
            return result
        elif choose == "numbers":
            #2.3. Các thuộc tính input kiểu số 
            print('Thuộc tính kiểu số:', df[lst_input].select_dtypes(include=['int','float']).columns)
            lists_ = df[lst_input].select_dtypes(include=['int32','float64','int','float']).columns
            result = pd.DataFrame(lists_, columns=['Thuộc tính kiểu số'])
            return result
        else:
            print("Please choose = [object or numbers]")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.2.  Xác định thuộc tính input là phân loại (Categorical field): Dùng để đưa các thuộc tính phân loại vào thuộc tính => Lấy dữ liệu là lst_phanloai
##### Mục đích xem lại các dữ liệu là thuộc tính thuộc categorical
'''
    parameter:
        - df: dataframe
        - lst_continious: thế hiện đó là kiểu dữ liệu
    Return
        - Dataframe theo loại dữ liệu
'''
def defineDtypeCategorical(df, lst_input, numbers = 20):
    try:
        categoricals = []
        continuous = []
        lst_phanloai = []

        for i in lst_input:    
            if df[i].dtypes =='object':
                lst_phanloai.append(i)
                i = '\'' + i + '\''
                categoricals.append(i)
            elif len(df[i].unique()) <= numbers and df[i].dtypes =='int':
                lst_phanloai.append(i)
                i = '\'' + i + '\''
                continuous.append(i)   
            else: pass    
        print('- Thuộc tính có thể phân loại kiểu chuỗi: ',', '.join(categoricals))
        print('- Thuộc tính có thể phân loại kiểu số: ',', '.join(continuous))
        return lst_phanloai
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.3. Xác định thuộc tính input là liên tục (Continious field): dùng để lựa chọn lại các thuộc tính lst_lientuc
##### Mục đích xem lại các dữ liệu là thuộc tính thuộc liên tục
'''
    parameter:
        - lst_input: phân loại dữ liệu lst_input
        - lst_phanloai: phân loại dữ liệu categorical
    return
        - lst_liên tục
'''
def defineDtypeContinuous(lst_input, lst_phanloai):
    try:
        lst_lientuc = list(set(lst_input) - set(lst_phanloai))
        t3 = []
        for i in lst_lientuc:
            i = '\'' + i + '\''
            t3.append(i)   
        print('- Thuộc tính liên tục:',', '.join(t3))
        return lst_lientuc
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.4. Xem lại thuộc tính phân loại: Xem lại các mức độ tần số xuất hiện dữ liệu
def categorical_value_counts_count(df, lst_categorical_choose):
    try:
        dicts = {}
        for i in lst_categorical_choose:
            dict_  = {i: list(df[i].sort_values().unique())}
            dicts.update(dict_)
        #result = pd.DataFrame.from_dict(dicts)
        return dicts
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

## B. PHÂN TÍCH BIẾN LIÊN TỤC (Continious field)
#########################################################################################
### 2.4. Các chỉ số mô tả thuộc tính output và input liên tục
##### Mục đích thực hiện xem các dữ liệu 'nameFeature', 'min', 'max', 'mean', 'median',
##### 'mode', 'std', 'var', 'kurtosis', 'text_kurtosis', 'skew', 'text_skew' => Xem thế nào và thực hiện tiếm các bước tiếp theo của biến continious
'''
    parameter:
        - df: data processing theo dataframe
        - lst_lientuc: continious output và input
'''
def summaryVariableContinuousAndCategorical(df, lst_lientuc, lst_output, columns = ['nameFeature', 'count', 'min', 'max', 'range', 'mean', 'median', \
        'mode', 'std', 'var', 'kurtosis', 'text_kurtosis', 'skew', 'text_skew']):
    try:    
        lst_ = lst_output+lst_lientuc
        results_ = []
        kurtosis_ = 0
        skew_ = 0
        for i in lst_:
            #print("\nGiá trị thống kê của", (lst_lientuc+lst_output)[i],":\n",stats.describe(df[(lst_lientuc+lst_output)[i]]))
            nameFeature = i
            count_ = round(df[i].count(),1)
            min_ = round(df[i].min(),1)
            max_ = round(df[i].max(),1)
            range_=round((max_ - min_),1)
            mean_ = round(df[i].mean(),1)
            median_ = round(df[i].median(),1)
            mode_ = round(df[i].mode()[0],1)
            std_ = round(df[i].std(),1)
            var_ = round(df[i].var(),1)
            kurtosis_ = round(kurtosis(df[i]),1)
            if kurtosis_ > 0:
                text_kurtosis = "Lệch phải"
            elif kurtosis_ < 0:
                text_kurtosis = "Lệch trái"
            elif kurtosis_ == 0:
                text_kurtosis = "Đối xứng"
            skew_ = round(skew(df[i]),1)
            if skew_ > 0:
                text_skew = "Nhọn"
            elif skew_ < 0:
                text_skew = "Bẹt"
            elif skew_ == 0:
                text_skew = "~Chuẩn"
            result = (nameFeature, count_, min_, max_, range_, mean_, median_, mode_, std_, var_, kurtosis_, text_kurtosis, skew_, text_skew)
            results_.append(list(result))
            kq = pd.DataFrame(results_, columns=columns)
        return kq
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.5. Biểu đồ phân phối => Mục đích: Xem mức độ thuộc tính xem mực độ phân bổ dữ liệu trong thuộc tính
'''
    parameter:
        - df: data pre-processing
        - lst_lientuc: những thuộc tính liên tục input
        - lst_output: những thuộc tín liên tục output
    return 
        - Biểu đồ phân tán dữ liệu
'''
def displotChart(df,lst_lientuc, lst_output, witdth=15, height=10, a=3, b=5):
    try:
        plt.figure(figsize=(witdth,height))
        n=0
        for i in (lst_lientuc+lst_output):
            n=n+1
            plt.subplot(a,b,n)
            sns.distplot(df[i])
        plt.tight_layout()
        return plt.show()
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.6. Biểu đồ phân phối chồng lên nhau: Xem mức độ các biên có sát nhau không gần nhau không
'''
    parameter:
        - df: data preprocessing
        - lst_lientuc: những thuộc tính liên tục input
        - lst_output: những thuộc tín liên tục output
    return
        - Biểu đồ phân tán giữa liệu
'''
def subplotsChart(df, lst_lientuc, lst_output, witdth=15, height=10):
    try:
        f, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(witdth,height))
        for i in (lst_lientuc+lst_output):
            sns.distplot(df[i], ax= ax1, hist=False)
        return plt.show()
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

## TÌM HIỂU VỀ OUTLIER
outlier = '''
    4. Phát hiện và xử lý ngoại lệ với outlier hợp lệ và xử lý outlier không hợp lệ
        4.1. Outlier ngoại lệ là gì ?
            - Outlier là 1 mẫu xuất hiện và tách xa khỏi tổng thể
        4.2. Phân loại outlier
            - Ngoại lệ đơn biến: có thể xem xét khi xem xét 1 biến duy nhất
            - Outlier hợp lệ: những outlier đúng với nghiệp vụ kinh tế
            - Outlier không hợp lệ: cần xem lại các nghiệp vụ kinh tế và loại bỏ chúng ra khỏi tập dữ liệu
        4.3. Nguyên nhân gây ra ngoại lệ ?
            - Tự nhiên
            - Nhân tạo: Lỗi do nhập liệu/ Lỗi đo lường/ Lỗi thử nghiệm/ Ngoại lệ có chủ ý/ Lỗi xử lý dữ liệu/ Lỗi lấy mẫu  
        4.4. Tác động của outlier đến thế nào với dữ liệu ?
            - Tăng phương sai lỗi và giám sức mạnh của kiểm định thống kê
            - Làm giảm quy tắc nếu ngoại lệ không phân bổ tự nhiên
            - Chung có thể bias hoặc ảnh hưởng đên mức độ dữ liệu được quan tâm
            - Chúng tác động đến giả định cơ bản của regression và giả định mô hình thống kê khác 
        4.5. Các phát hiện outlier
            - Dùng boxlot, histogram, scatter plot
            - Quy tắc ngón tay
            - Phạm vi giá trị: Q1-1.5*IQR và Q1+1.5*IQR
            - Phạm vi phân vị từ 5 đến 95 nằm ngoài được coi là outlier
            - Phát hiện theo nghiệp vụ bai toán
        4.6. Cách loại bỏ outlier
            - Xóa bỏ: Mức độ ảnh đến bộ mẫu => Xóa khi outlier bất hợp lý/ Hợp lệ thi không xóa => Chỉ xóa những outlier có ngoại lệ nhỏ về số lượng
            - Điện giá trị thay thế bằng: mode/ median/ mean
            - Biến đổi chúng bằng các giá trị thông qua hàm với hệ số e =>Giảm giá trị về cùng cơ số, 
            phương thức log để tạo ra cột để độ lớn scale lại và tiệt tiêu độ lớn đó
            - Tách riêng bộ dữ liệu outlier và bộ không có outlier => kỷ thuật: Treat separately
            - Thuật toán chấp nhận outlier nhưng chỉ chấp nhận outlier hợp lệ cho nên cần xử lý triển để outlier không hợp lệ
            - Thuật toán Cây quyết định (Decision Tree) cho phép xử lý tốt các ngoại lệ do việc tạo biến
        4.7. Có thể nhìn thông qua để biết xem biết và phát hiện xử lý ngoại lệ
            - Mean/Median/Mode => Giá trị phải ngang ngang nhau => nếu không thì mean sẽ trội hơn so với thuộc tính median và mode
            - Rất quan trọng để điều tra bản chất của ngoại lệ trước khi quyệt định
'''
### 2.7. Biểu đồ boxplot dùng xem outlier => Xem mức độ dữ liệu phân bổ dữ liệu outlier hợp lý và không hợp lý
'''
    parameter
        - df: data preprocessing
        - lst_lientuc: những thuộc tính liên tục input
        - lst_output: những thuộc tín liên tục output
    return 
        - về biểu đô outlier dữ liệu của từng thuộc tính
'''
def boxplotChart(df, lst_lientuc, lst_output, a=3, b=5):
    try:
        plt.figure(figsize=(15,8))
        n=0
        for i in (lst_lientuc+lst_output):
            n=n+1
            plt.subplot(a,b,n)
            sns.boxplot(df[i])
        plt.tight_layout()
        return plt.show()
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.8. Xem xét outlier (thuộc tính 'compression_ratio'): Hiển thị những giá trị outlier của thuộc tính
##### filterOutlier: thuộc tính cần tìm hiểu về outlier
##### listsDisplay: Hiện thị các thuộc tính kèm theo
'''
    parameter
        - df: data preprocessing
        - lst_lientuc: những thuộc tính liên tục input
        - lst_output: những thuộc tín liên tục output
        - Dùng 1.5 kaf theo kiểu IQR và hay nó là boxplot
        - Con dùng 3 theo nguyên lý z_score
    return 
        - dataframe các dữ liệu đến outliers
'''
def filterOutlier(df, lst_input, lst_output, columns=['name_feature', 'IQR', 'Q1', 'Q3', 'cận dưới', 'cận trên', \
                                                        'number_total_oulier', 'outliers_per_%','count', 'min', 'max', 'range', 'mean', 'median', \
                                                        'mode', 'std', 'var', 'kurtosis', 'text_kurtosis', 'skew', 'text_skew']):
    try:
        lst_continuous = lst_output + lst_input
        results_ = []
        for i in lst_continuous:
            count_ = round(df[i].count(),1)
            min_ = round(df[i].min(),1)
            max_ = round(df[i].max(),1)
            range_=round((max_ - min_),1)
            mean_ = round(df[i].mean(),1)
            median_ = round(df[i].median(),1)
            mode_ = round(df[i].mode()[0],1)
            std_ = round(df[i].std(),1)
            var_ = round(df[i].var(),1)
            skew_ = round(skew(df[i]),1)
            kurtosis_ = round(kurtosis(df[i]),1)
            if kurtosis_ > 0:
                text_kurtosis = "Lệch phải"
            elif kurtosis_ < 0:
                text_kurtosis = "Lệch trái"
            elif kurtosis_ == 0:
                text_kurtosis = "Đối xứng"
            skew_ = round(skew(df[i]),1)
            if skew_ > 0:
                text_skew = "Nhọn"
            elif skew_ < 0:
                text_skew = "Bẹt"
            elif skew_ == 0:
                text_skew = "~Chuẩn"
            IQR_= scipy.stats.iqr(df[i])
            Q3_ = np.quantile(df[i].dropna(), 0.75)
            Q1_ = np.quantile(df[i].dropna(), 0.25)
            number_outlier_upper = df.loc[df[i] > Q3_ + 1.5 * IQR_].shape[0]
            number_outlier_lower = df.loc[df[i] < Q1_ - 1.5 * IQR_].shape[0]
            number_total_oulier = number_outlier_lower + number_outlier_upper
            outliers_per_ = (number_total_oulier)/df.shape[0]
            result = (i,round(IQR_,2), round(Q1_,2), round(Q3_,2), round(number_outlier_lower,2),  round(number_outlier_upper,2),\
                  round(number_total_oulier,2),  round(outliers_per_*100,2), count_, min_, max_, range_, \
                      mean_, median_, mode_, std_, var_, kurtosis_, text_kurtosis, skew_, text_skew)
            results_.append(list(result))
            kq = pd.DataFrame(results_, columns=columns)
        return kq
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 4.2. Xem xét outlier (thuộc tính 'compression_ratio'): Hiển thị những giá trị outlier của thuộc tính
##### filterOutlier: thuộc tính cần tìm hiểu về outlier
##### listsDisplay: Hiện thị các thuộc tính kèm theo
##### Q1 - 1.5 * IQR to Q3 + 1.5 * IQR
def filter_outlier_dataframe(df, filterOutlier):
    try:
        IQR= scipy.stats.iqr(df[filterOutlier])
        Q3 = np.quantile(df[filterOutlier].dropna(), 0.75)
        Q1 = np.quantile(df[filterOutlier].dropna(), 0.25)
        result1 = df.loc[df[filterOutlier] > Q3 + 1.5 * IQR]
        result2 = df.loc[df[filterOutlier] < Q1 - 1.5 * IQR]
        result = pd.concat([result1, result2])
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 4.3. Filter dataframe theo thuộc tính
        
def filter_dataframe(df, feature_name, name_type):
    try:
        result = df.loc[df[feature_name] == name_type]
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 4.2. Xem xét outlier (thuộc tính 'compression_ratio'): Hiển thị những giá trị outlier của thuộc tính
##### filterOutlier: thuộc tính cần tìm hiểu về outlier
##### listsDisplay: Hiện thị các thuộc tính kèm theo
##### Q1 - 1.5 * IQR to Q3 + 1.5 * IQR
def filter_not_outlier_dataframe(df, filterOutlier):
    try:
        IQR= scipy.stats.iqr(df[filterOutlier])
        Q3 = np.quantile(df[filterOutlier].dropna(), 0.75)
        Q1 = np.quantile(df[filterOutlier].dropna(), 0.25)
        result1 = df.loc[df[filterOutlier] <= Q3 + 1.5 * IQR]
        result2 = df.loc[df[filterOutlier] >= Q1 - 1.5 * IQR]
        result = pd.concat([result1, result2])
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

## C. PHÂN TÍCH BIẾN PHÂN LOẠI (Categorical field)
###########################################################################
### 2.9. Các chỉ số mô tả thuộc tính: xem lại mức tần số mức độ dữ liệu
'''
    parameter:
        - df: dataframe - preprocessing
        - lst_category_number: kiểu categorical những là số
        - lst_category_object: là kiểu dữ liệu object cần phải standardization
        - columns: thể hiện số phần tử tập
    return
        - Trả về print kết quả và dataframe dữ liệu số phần tử
'''
def summaryCategorical(df, lst_category_number, lst_category_object, columns = ['feature','Số phân tử']):
    try:
        lsts_ = []
        lst_categorical = lst_category_number + lst_category_object
        for i in lst_categorical:
            nameFeature = i
            values = len(df[i].unique())
            value_counts = df[i].value_counts()
            print('-', '\033[4m'+'Mô tả biến ', i+'\033[0m', ':',len(df[i].unique()),'giá trị')
            print(value_counts)
            lst_ = (nameFeature, values)
            lsts_.append(list(lst_))
            kq = pd.DataFrame(lsts_, columns=columns)
        return kq
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")


### 2.10. Value_count theo từng thuộc tính: xem mức độ các biên, count các phần tử
'''
    parameter:
        - df: dataframe-pre-processing
        - lst_category_number: kiểu categorical những là số
        - lst_category_object: là kiểu dữ liệu object cần phải standardization
    return
        - trả về 1 dataframe
'''
def categorical_value_counts(df, lst_categorical):
    try:
        #lst_categorical = lst_category_number + lst_category_object
        value_counts_ = df[lst_categorical].value_counts()
        kq = pd.DataFrame(value_counts_)
        return kq
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")   

### 2.11. Xem barchart xem mức độ dữ liệu
'''
    parameter:
        - df: dataframe-pre-processing
        - lst_category_number: kiểu categorical những là số
        - lst_category_object: là kiểu dữ liệu object cần phải standardization
    return
        - trả về 1 visualization
'''
def barchart(df, lst_category_number, lst_category_object, width=15, height=10, rotation=45, a =4, b=3):
    try:
        # Biểu đồ barchart
        plt.figure(figsize=(width,height))
        n=0
        lst_categorical = lst_category_number + lst_category_object
        for i in lst_categorical:
            n=n+1
            plt.subplot(a,b,n)
            df[i].value_counts().plot.bar()
            plt.title(i)
            plt.xticks(rotation=rotation)

        plt.tight_layout()
        return plt.show()
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")
###################################################################################################################################################################
#                                                               BI-VARIABLE ANALYSIS _ PHÂN TÍCH 2 BIẾN                                                           #
###################################################################################################################################################################
import pandas as pd 
from pandas.core.frame import DataFrame
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

BI_variable_analysis = '''
    3. Phân tích 2 biến (BI-variable analysis
        3.0. Phân tích 2 biền làm gì ?
            - Tìm sự liên kết (association) và không liên kết (Disassociation)
                * Hai biến input có độc lập với nhau không ?
                * Hai biến input và output có phụ thuộc lãnh nhau không 
                => Có sự liên kết với nhau không ? 
                => Nếu phú thuộc thì quá tốt với model
                * Thực tế các biến input khó có thể độc lập với nhau 
                => Cần xem mức độ tương quan giữa các biến
                * Các biến input phụ thuộc nhau các ít càng tốt
                * Biến input luôn phải phụ thuộc với biến output thì mới có ý nghĩa
        3.1. Biến liên tục với biến liên tục (Continuous vs Continuous)
            - Dùng biểu đồ phân tán
                * Đó là cách phù hợp để xem mối quan hệ 2 biến
                * Cho thấy mỗi quan hệ là tuyến tính hoặc phi tuyến tính
            - Dùng Correction
                * Tương quan khác nhau giữa -1 đến 1
                * Bé hơn 0.3 tương quan rời rác
                * Từ 0.3 đến 0.6 tương quan
                * Lớn hơn 0.6 tương quan mạnh mẽ
        3.2. Hai biến phân loại (Categorical vs Categorical)
            Step 1: Dùng two-way table
                - Bắt đầu phân tích mỗi quan hệ bằng cách tạo 2 chiều Count
                - Các dòng category theo các dòng khác nhau
            Step 2: Stacked column chart - Trực quan hóa 2 cột chồng lên nhau
            Bước 3: Dùng Chi-square
                - Kiểm định 2 biến độc lập hay phụ thuộc
                    * Thức đo giá trị thống kê: Statistic >= Critical value: Dữ liệu độc lập, ngược lại là dữ liệu phụ thuộc
                    * 2. Theo giá trị p-value: p-value <= alpha: 2 biến độc lập, ngược lại 2 biến phụ thuộc hoặc alpha = 1-prod trong đó thường prod: 0.95
        3.3. Biến liên tục vs biến phân loại (Categorical vs Continuous) => Một trường hợp phức tạp
            - Trực quan hóa dữ liệu bằng boxlot
                * Với số lượng thuộc tính ít, không hiển thị ý nghĩa thống kê
            - Dùng ANOVA - để xem mức độ tương quan: aov_table[aov_table['PR(>F)'] < alpha] => Có tương quan với nhau 
        3.4. Mục đích phân tích để làm gì ?
            Thứ 1: Phát hiện missing value và outlier
            Thứ 2: Chọn các thuộc tính categorical, continious phù hợp với bài toán
            Thứ 3: Xem có thể thực các bước tiếp theo      
'''
##############################################################################################################
## A. PHÂN TÍCH 2 BIẾN CONTINIOUS VS CONTINIOUS
### 3.1. Ma trận hệ số tương quan: dùng để xem xét mức độ tương quan của 2 biết liên tục => Quan sát nhìn qua
'''
    dataframe:
        - df: dataframe - pre-processing
        - lst_output: biến output numbers có tính liên tục
        - lst_lientuc: biến input numbers có tính liên tục
    return
        - correction => để hiểu đc dữa biến liên tục và biến 
'''
def corr_continious(df, lst_output_continious, lst_input_continious):
    try:
        return df[lst_output_continious + lst_input_continious].corr()
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.2. Biểu đồ tương quan: Thể hiện tương quan giữa các biên nhưng rất khó nhìn hạn chế dùng
'''
    parameter:
        - df: dataframe - data pre processing
        - lst_output_continious: biến output đâu ra
        - lst_input_continious: biến input
    return
        - Visualization thể hiện nhưng tính không hiệu quả lắm
'''
def pairplot_chart(df, lst_continious):
    try:
        return sns.pairplot(df[lst_continious], corner = True)
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

###3.3. Biến inputs liên tục với biến output liên tục: Dùng để xem mực độ tương quan của biến output và output
##### Khi nào biến output là biến continious
##### Biến output và input phải có mức độ tương quan với nhau
'''
    parameter:
        - df: dataframe - pre-processing
        - lst_output_continious: biến duy nhất về output
        - lst_input_continious: biến input
        - strong: tương quan mạnh
        - correlate: Có tương quan
    return
        -  Trả về dataframe
'''
def correlate_output_two_continious(df, feature_output_continious, lst_input_continious, strong=0.6, correlate=0.3):
    try:
        j = feature_output_continious
        pair1 = []
        pair2 = []
        pair3 = []

        for i in lst_input_continious:
            corr = abs((df[[i,j]].corr().loc[[i],[j]]).values[0][0])
            if corr >= strong:
                pair1.append(['strong',j,i,  round(corr,2)])
            elif corr >= correlate:
                pair2.append(['correlate',j,i,  round(corr,2)])
            else: 
                pair3.append(['weak',j,i,  round(corr,2)])
        kq1 = pd.DataFrame(pair1, columns=['correlate','Variable_output', 'Variable_input',  "corr"])
        kq2 = pd.DataFrame(pair2, columns=['correlate','Variable_output', 'Variable_input',  "corr"])
        kq3 = pd.DataFrame(pair3, columns=['correlate','Variable_output', 'Variable_input',  "corr"])
        result = kq1.append(kq2)
        result = result.append(kq3)
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

###3.4. Tương quan giữa các biến inputs liên tục: Xem mức độ tương quan của các thuộc tính input liên tục với nhau
##### Biến input càng tương quan thì cần xem xét nên chọn biến nào đề phù hợp với bài toán
'''
    parameter:
        - df: dataframe - pre-processing
        - lst_input_continious: biến input
        - choose: [strong, medium, correlate, weak, all]
'''
def correlate_input_two_continious(df, lst_input_continious, choose="strong"):
    try:
        pair1 = []
        pair2 = []
        pair3 = []
        pair4 = []

        for i in lst_input_continious:
            for j in lst_input_continious[lst_input_continious.index(i)+1:]:
                corr = abs((df[[i,j]].corr().loc[[i],[j]]).values[0][0])
                if corr >= 0.9:
                    pair1.append(['strong',j,i, round(corr,2)])         
                elif corr >= 0.6:
                    pair2.append(['medium',j,i, round(corr,2)])          
                    
                elif corr >= 0.3:
                    pair3.append(['correlate',j,i, round(corr,2)])
                else: 
                    pair4.append(['weak',j,i, round(corr,2)])
        if choose =="strong":
            result1 = pd.DataFrame(pair1, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            return result1
        elif choose == "medium":
            result2 = pd.DataFrame(pair2, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            return result2
        elif choose =="correlate":
            result3 = pd.DataFrame(pair3, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            return result3
        elif choose == "weak":
            result4 = pd.DataFrame(pair4, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            return result4
        elif choose == "all":
            result1 = pd.DataFrame(pair1, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            result2 = pd.DataFrame(pair2, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            result3 = pd.DataFrame(pair3, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            result4 = pd.DataFrame(pair4, columns=['correlate','Variable_input1', 'Variable_input2',  "corr"])
            result = result1.append(result2)
            result = result.append(result3)
            result = result.append(result4)
            return result
        else:
            print("Please check and call to me")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

################################################################################################################################
## B. PHÂN TÍCH BIẾN LIÊN TỤC VÀ BIẾN PHÂN LOẠI (Continious & Categorical)
### 3.5. Giữa biến output liên tục và biến inputs phân loại: thể hiện mức độ tương quan giữa các biến continious vs categorical có phụ thuộc nhau
##### Trường hợp là input vs input => Tương quan càng mạnh thì cần loại bỏ
##### Trường hợp là input vs output => Tương quan càng mạnh thì lấy biến input
##### Nên việc chọn các thuộc tính trong bài toán cực kỳ quan trọng
##### Có 3 kết quả để choose trong [Kết quả, Hiển thị ANOVA, Tương quan]
'''
parameter:
    - df: data pre-processing
    - lst_categorical: lst_category
    - feature_variable: Biến liên tục
return
    - Trả về choose [all, anova, correlate, result]
'''
def correlate_output_continious_categorical(df, lst_categorical, feature_variable, choose ="correlate_output"):
    try:
        ## Tạo chuỗi
        prob = 0.95
        alpha = 1.0 - prob
        string = []
        for i in lst_categorical:
            t = 'C(' + i + ')'
            string.append(t)
        string = '+'.join(string)
        ## ANOVA và chọn biến inputs phân loại có ảnh hưởng lớn đến biến output
        model = ols(feature_variable+' ~ '+ string, data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        aov_table1 = aov_table[aov_table['PR(>F)'] < alpha]
        lst_phanloai_chosen = aov_table1.index.str.extract('(C\()(\w+)')[1].to_list()
        if choose == "all":
            results = [print('- Kết luận: Các thuộc tính phân loại có ảnh hướng đáng kể dến thuộc tính output (price):',
            lst_phanloai_chosen), print('Các thuộc tính có ảnh hướng đáng kể dến thuộc tính price:', '\n', aov_table1),
            print('Kiểm định ANONA giữa thuộc tính price và các thuộc tính phân loại:','\n', aov_table) ]
            return results
        elif choose == "anova":
            return aov_table
        elif choose == "correlate_output":
            return aov_table1
        elif choose == "result":
            results = DataFrame(lst_phanloai_chosen, columns=["Ảnh hưởng đến biến continious "+feature_variable])
            return results
        else:
            print("Please check again")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")
    
### 3.6. Giữa các biến inputs phân loại và biến inputs liên tục
#### Xem mức độ tương quan của các biên phân loại vs liên tục theo 2 cách choose: [matrix, melt]
#### Matrix: thể hiện 2 biến dòng và côtj
#### melt: thể hiện theo hàng cột
'''
    parameter:
        - df: dataframe - pre-processing
        - lst_continious: list giá trị liên tục
        - lst_categorical: list giá trị phân loại
        - choose="matrix" 
            matrix: thể hiện 2 biến dòng và côt
            melt: thể hiện theo hàng cột
    return
        - Trả về kết quả là matrix hay melt
'''
def correlate_input_variable_continious_inputCategorical(df, lst_continious, lst_categorical, choose="matrix"):
    try:
        ## Tạo chuỗi
        prob = 0.95
        alpha = 1.0 - prob
        names = {}
        for j in lst_continious:
            string = []
            name = "aovTable_"+j
            for i in lst_categorical:
                t = 'C(' + i + ')'
                string.append(t)
            string = j + ' ~ ' + '+'.join(string)
            string

            model = ols(string, data=df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            aov_table1 = aov_table[aov_table['PR(>F)'] < alpha]
            name = {j: i, j: round(aov_table1['PR(>F)'],2)}
            names.update(name)
        if choose == "melt":
            result_melt = pd.DataFrame.from_dict(names).melt()
            return result_melt
        elif choose == "matrix":
            result = pd.DataFrame.from_dict(names)
            return result
        else:
            print("Check para choose = [melt or matrix]")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

########################################################################################################################
## C. PHÂN TÍCH 2 BIẾN PHÂN LOẠI
### 3.7 Biến phân loại và Biến phân loại (Categorical & Categorical)
##### Kiểm tra mức độ tương quan của 2 biến categorical: Phụ thuộc hay độc lập
##### Chọn được các trường hợp: choose [Phụ thuộc, Độc lập, Tất cả]
'''
    parameter:
        - df: dataframe - preprocessing
        - lst_categorical: biến categorical
        - choose: [dependent=Phụ thuộc/ independent=Không phụ thuộc/ All=tất cả ]
    return
        - Trả về dataframe
'''
def correlate_2_variable_categorical(df, lst_categorical, choose ="all"):
    try:
        prob = 0.95
        alpha = 1.0 - prob

        pair1 = []
        pair2 = []

        for i in lst_categorical:
            for j in lst_categorical[lst_categorical.index(i)+1:]:
                crosstab = pd.crosstab(df[i], df[j])
                stat, p, dof, expected = chi2_contingency(crosstab)
                #critical = chi2.ppf(prob, dof)
                if p <= alpha:
                    pair1.append(["Phụ thuộc nhau (reject H0)",i,j])
                else:
                    pair2.append(["Độc lập nhau (fail to reject H0)",i,j])
            
        if choose =="dependent":
            kq1 = pd.DataFrame(pair1, columns=['Tuong_quan','Variable_input1', 'Variable_input2'])
            return kq1
        elif choose == "independent":
            kq2 = pd.DataFrame(pair2, columns=['Tuong_quan','Variable_input1', 'Variable_input2'])
            return kq2
        elif choose == "all":
            kq1 = pd.DataFrame(pair1, columns=['Tuong_quan','Variable_input1', 'Variable_input2'])
            kq2 = pd.DataFrame(pair2, columns=['Tuong_quan','Variable_input1', 'Variable_input2'])
            result = kq1.append(kq2)
            return result.drop_duplicates()
        else:
            print("Please choose = [Phụ thuộc or Độc lập or Tất cả]")
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

###################################################################################################################################################################
#                                                            CLEANING DATA TIDYING - LAM SÁCH DỮ LIỆU VÀ THU GỌN DỮ LIỆU                                          #
###################################################################################################################################################################
Exploring_data = '''
    CLEANING DATA - LÀM SẠCH DỮ LIỆU
    1. Khám phá dữ liệu (Exploring data)
        Step1: Khám phá dữ liệu
            - Khám phá dữ liệu để làm gì ?
                * Chẩn đoán các vấn để như ngoại lệ, giá trị bị thiếu và trung lặp dữ liệu
            - Hiểu ý nghĩa của các thuộc tính và biến
            - Phát hiện các lỗi bên trong càng nhiều càng tốt, các lỗi tiềm ẩn bên trong 
            => Xây dựng 1 bộ dữ liệu phù hợp với yêu cầu bài toán
            - Các đánh giá nhanh nhất: thông qua các công cụ trực quan
            - Dữ liệu trùng lặp: trùng lặp trên 1 tập hợp các thuộc tính hữu hạn
        Step2: Chuẩn hóa dữ liệu
            - Làm sách dữ liệu xem dữ liệu bên trong có vấn đề gì không ?
            - Dữ liệu ban đầu luôn bị lỗi
            - Cần chuẩn hóa dữ liệu để tìm ra các vấn đề cần giải quyết
        Step3: Một số vấn đề thường gặp
            - Tên các cột không thống nhất
            - Dữ liệu bị thiếu: missing value, null, thiệu thuộc tính, thiếu thông tin, ký tự 
            => Thiếu 1 số thuộc tính cần thiết của bài toán
            - Ngoại lệ (outlier): 
                * Hợp lý không được xóa cần xem xét lại nghiệp vụ
                * Không hợp lệ: nếu xóa thì xem lại tỉ lệ bộ mẫu nếu tỉ lệ thấp thì xóa dữ liệu, 
                nếu cao thì cần xem xét lại tổng thệ của bộ mẫu
            - Dòng dữ liệu trùng lặp
                * Xem dữ liệu trùng lặp trên 1 bộ thuộc tính 
                => Nên xóa dữ liệu trung lặp
            - Dữ liệu không gọn gàng cần nhóm lại
            - Một cột chứa nhiều thông tin hoặc có những thông tin không mong muốn
        Step4: Quan trọng trong kham phá dữ liệu
            - Tần số đếm: đếm giá trị duy nhất trong dữ liệu => Dùng value_count() để đếm
            - Kiểu dữ liệu các cột đã đúng chưa
            - Vẽ biểu đồ boxlot xem xét tỉ lệ outlier nhiều hay ít
        Step5: Trực quan dữ liệu khám phá
            - Sử dụng biểu đồ phù hợp để phát hiện lỗi nhanh hơn
            - Tìm những điểm chung các pattern trong dữ liệu
            - Lập kế hoạch các bước để làm sạch dữ liệu
            - Dùng 1 số biểu đồ để khám phá dữ liệu
                * Bar plot: Biểu diễn số lượng rời rạc
                * Histogram: Biễu diễn min, max, xem tần số
                * Boxplot: biểu diễn min, max, 1Q, median, 3Q, outliers
                * Scatter plot: kiểm định ứng dụng vào model, phát hiện thêm outlier của 2 biến có hợp lý hay không ?
        Step6: Xác định error
            - Không phải các outlier cũng hợp lý và không hợp lý cần xem xét
            - Một số có thể là lỗi, có thể 1 số khác có giá trị là hợp lệ
    => Đã năm ở B1 đến B5
'''
Tidying_data = '''
    2. Thu gọn dữ liệu
        2.1. Mục tiêu
           - Đây không phải là phân tích các bộ dữ liệu mà là chuẩn bị chung theo cách chuẩn hóa dữ liệu phù hợp trước khi phân tích
        2.2. Một số loại dữ liệu lộn xộn cần phải giải quyết
            - Tiêu đề cột là giá trị, không phải tên biến
            - Nhiều biến được lưu trữ trong 1 cột
            - Các biến được lưu trữ trong một cột
            - Nhiều đơn vị mẫu được lưu trữ trong 1 bảng
            - Một mẫu quan sát duy nhất được lưu trữ trong nhiều bảng và file
        2.3. Dữ liệu như thế nào là dữ liệu gọn ?
            - Kết quả của quá trình thu gọn dữ liệu
            - Dễ dàng thao tác, mô hình hóa và trực quan
            - Mỗi tập dữ liệu gọn gàng được sắp sếp sao cho mỗi biến là một cột và mỗi quan sát là 1 hàng
        2.4. Đặc điểm đo lường
            - Một biến đo lường phải ở trong 1 một cột
            - Mỗi mẫu khác nhau của biến đó nên ở một hàng khác nhau
            - Cần mỗi biến là 1 cột khác nhau
            - Nếu có nhiều bảng, thì chúng có 1 cột trong bảng cho phép chung liên kết
        2.5. Pivoting data (un-melting data)
            - Trong melting data chung ta chuyển các cột thành các dòng
            - Trong pivoting data: chuyển các giá trị duy nhất thành các cột riêng biệt
            - Dùng để tạo các báo cáo
            - Vi phạm nguyên tắc của tity data: các dòng chứa các mẫu
            - Sử dụng pivot hoặc pivottable 
'''
combining_data = '''
    3. Kết hợp dữ liệu
        - Vấn đề
            * Dữ liệu không phải lúc nào cung lưu trong 1 tệp lớn 
        - Ưu điểm
            * Dễ dàng lưu trữ và chia sẽ
            * Có thể lưu trữ mỗi ngày
            * Có thể kết hợp với nhau để làm sạch dữ liệu hoặc ngược lại
        - Giải pháp
            * Phương pháp concat
                + Nối dữ liệu
                + Sử dụng để kết nối dataframe cùng cấu trúc dữ liệu
                + Nối nhiều tập tinh nhiều file
            * Phương pháp trộn dữ liệu merge()
                + Tương tự như phép join bằng bảng CSDL
                + Kết hợp các bộ khác nhau dựa trên các cột chung hay key
'''
cleaning_data = '''
    4. Làm sạch dữ liệu (Cleaning data)
        Step1: Kiểu dữ liệu (data type)
            - Theo yêu cầu phải chuyển dữ liệu từ kiểu này sang kiểu khác => Cột số có thể chứa chuổi hoặc ngược lại
            - Chuyển dữ liệu thành số pd.to_numeric(df['tên_cột']
            - Chuyển đổi kiểu dữ liệu cho cột: df['tên_cột'].astype(kiểu_dữ_liêu)
        Step2: Dữ liệu phân loại (Categorical data)
            - Chuyển categorical data chuyển dữ liệu thành 'category' dtype 
            - Có thể làm cho dataFrame giảm được kích thước trong memory
            - Có thể làm cho chúng được sử dụng dễ dàng bởi các thư viện python khác
        Step3: Thao tác trên chuỗi
            - Phần lớn việc làm sạch dữ liệu liên quan đến thao tác chuỗi
            - Hầu hết dữ liệu trên thế giới là văn bản không có cấu trúc
            - Có rất nhiều thư viện hỗ trợ built-in và thư viện bên ngoài
            - Các công cụ xử lý chuỗi
                * Sử dụng regular expression
                * Xử lý các dữ liệu không phù hợp
                * Xử lý dữ liệu trùng lặp
                * Xử lý dữ liệu thiếu (missing value)
        Mục đích:
            - Các function: df.apply()/ Regular expression: re.compile()/ User defined function

'''

import pandas as pd 
import numpy as np
import warnings
import pandas_profiling as pp # tổng quan ban đầu về dữ liệu => Cài trên này
warnings.filterwarnings('ignore')
import glob

## A. CLEANING DATA - LÀM SẠCH DỮ LIỆU
### 3.1. Chuyển cột chứa giá trị thay vi chứa biến nên dùng melt
##### Melt df into new dataFrame
##### Mục đích: Chuyển dữ liệu dạng báo cáo về dạng cột để thực hiện phân tích dữ liệu
'''

'''
def change_columns_to_rows(df, id_vars, var_name, value_name):
    try:
        df_melted = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)
        return df_melted
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.2. Short value => Mục đích short giá trị hoặc short theo biến
'''

'''
def sort_feature(df, by=[]):
    try:
        df_sort = df.sort_values(by=by)
        return df_sort
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.3. Merge in Datatidying
##### Dùng hàm merge để thực hiện kết nối 2 bảng
'''

'''
def merge_data(df1, df2, how = 'inner', on = []):
    try:
        result = pd.merge(df1, df2, how=how, on=on)
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.4. Dữ liệu trên nhiều file mục đích ghép nhiều file lại nhiều
'''

'''
def load_mult_file(dir_name_file="", names="", index_col=None, header =0):
    try:
        allFiles = glob.glob(dir_name_file)
        frame = pd.DataFrame()
        df_list = []
        for file_ in allFiles:
            if dir_name_file.endswith("csv"):
                df = pd.read_csv(dir_name_file, names=names, index_col=index_col, header=header)
            elif dir_name_file.endswith("xlsx"):
                df = pd.read_excel(dir_name_file, names=names, index_col=index_col, header=header)
            elif dir_name_file.endswith("json"):
                df = pd.read_json(dir_name_file, names=names, index_col=index_col, header=header)
            else:
                print("Please see file and path")
            df.columns = map(str.lower, df.columns)
            df_list.append(df)
        results = pd.concat(df_list)
        return results
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

# Tạo DataFrame chỉ gồm Biến output, Biến intputs được chọn (phân loại và liên tục)
'''

'''
def combiningData(df, lst_phanloai_chosen, lst_lientuc_chosen, lst_output):
    try:
        data0 = df[lst_phanloai_chosen+lst_lientuc_chosen+lst_output]
        results = data0.drop_duplicates().reset_index()
        return results
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

#############################################################################################################################################################
#                                                       DATA STANDARDIZATION - CHUẨN HÓA DỮ LIỆU                                                            #
#############################################################################################################################################################


DATA_STANDARDIZATION_CHUAN_HOA_DU_LIEU = '''
    1. Nhìn chung
        1.1. Lợi ích của việc chuẩn hóa dữ liệu
            - Sử dụng phân tích và sử dụng dữ liệu một cách nhất quán
        1.2. Có nên chuẩn hóa dữ liệu không ?
            - Biết được điểm mạnh và điểm yếu
            - Thu thập dữ liệu sao phù hợp
            - Dọn dẹp và phân tích dữ liệu phù hợp
        1.3. Khi nào cần chuẩn hóa dữ liệu ?
            - Khi xây dựng model
                * Mô hình không gian tuyến tính
                * Các tính năng dữ liệu có phương sai cao
                * Các tính năng dữ liệu liên tục có đơn vị đo lường khác nhau
                * Giả định tuyến tính khác nhau
            - Nhu cầu chuyển đổi dữ liệu có thể phụ thuộc vào mô hình hóa mà chúng ta dự định sử dụng
                * Phương sai của biến đâu ra không thay đổi khi chung ta thay đổi biến đầu vào
                * Đảm báo rằng mối quan hệ giữa các biến đầu vào và biến đầu ra xấp xỉ tuyến tính
    2. Log normalization
        2.1. Ý nghĩa
            - Mục đích: dùng log để thay đổi hình dạng phân phối của biến trên biểu đồ phân phối
            - Được sử dụng để làm độ lệch phải (right skewness) của các biến:
                * Chỉ áp dụng với số dương
                * Không áp dụng được với phân phối lệch trái
                * Nếu sử dụng log may mắn thì sẽ trả về phân phối chuẩn kết quả hiệu suất cao hơn
        2.2. Cách áp dụng hàm log
            - Chuyển đổi log tự nhiên
            - Log tự nhiên có cơ số e = 2.718
            - Năm bắt những thay đổi tương đối, cương độ thay đổi, cường độ thay đổi và dữ mọi thứ trong không gian dương
            - Chia dữ liệu trong phạm vi theo số log
            - Dùng cho skewed và wedi Distribution
        2.3. Nên sử dụng khi nào ?
            - Phân phối lệch phải và độ lệch dương => skew trả về dương
            - Nếu thành phân theo cấp số nhân => sử dụng log
            - Dữ liệu được phân loại theo tứ tự độ lớn
            - Phương sai lớn và phân phối lệch phải thì sử dụng log
            - Log áp dụng để giảm thiểu outlier
            - Áp dụng trên các thuộc tính riêng lẽ
    3. Feature scaling
        3.1. Feature scaling và ý nghĩa
            - Là 1 công việc chuẩn hóa dữ liệu
            - Áp dụng các tính năng và các biến dữ liệu
            - Chuẩn hóa dữ liệu trong 1 phạm vi cụ thể
            - Giúp tăng tốc tính toán trong một thuật toán
            - Áp dụng cho các biến thuộc tính đầu vào dữ liệu chỉ áp dụng thuộc tính kiểu số
            - Không scale các thuộc tính categorical
            - Chỉ scale thuộc tính kiểu dữ liệu continuous
            - Giúp thuộc tính giảm cường độ/ trọng số
        3.2. Đặc điểm
            - Feature scaling giúp cân bằng các tính năng hoặc thuộc tính là như nhau
            - Sự công băng giữa các thuộc tính giữa liệu
            -=> Khi làm dữ liệu thì sẽ làm dữ liệu gốc trước, sau đó mới scale dữ liệu để chọn model tốt nhất
        3.3. Các thuật toán thường chuẩn hóa dữ liệu/ Thuật toán không áp dụng
            - Áp dụng cho các thuật toán
                * PCA: Cố gắng giúp các tính năng có phương sai tốt nhất
                * Gradient Descent: Tốc độ tính toán tăng khi tính toán Theta trở nên nhanh hơn sau khi scaling
                * K-nearest neighbors: KNN sử dụng đo khoảng cách Euclidean, nếu chúng ta muốn tất cả các thuộc tính có đóng góp như nhau
                * K-means: Các cụm cũng được xem xét gần nhau theo khoảng cách như KNN
                * Khi thực hiện thuật toán ML => Nếu có khoảng cách thì chúng ta cần phải chuẩn hóa, con không thì chung ta không cần chuẩn hóa
            - Không áp dụng cho các thuật toán
                * Naive Bayer 
                * Linear Discriminant Analysis
                * Tree-Based models
                => Những thuật toán không ảnh hưởng bởi feature scaling
                => Các thuật toán không dựa trên khoảng cách thì không cần phải áp dụng feature scaling
        3.4. Phương pháp dùng Feature Scaling
            3.4.1. StandardScaler
                - Một kỹ thuật hữu ích để biến đổi các thuộc tính có phân phối Gaussian và các giá trị trung bình (mean) & 
                độ lệch chuẩn (std) khác nhau thành phân phối Gausian tiêu chuẩn với giá trị trung bình là 0 và 
                độ lệch chuẩn là 1 và tuân theo công thức sau cho mỗi tính năng/ thuộc tính
                - Công thức: (xi -mean(x))/stdev(x)
                - Được ưu tiên sử dụng khi các thuộc tính của chung ta có phân phôi chuẩn hoặc xấp xỉ chuẩn (~0)thì chung ta ưu tiền dùng StandardScaler 
                => chuyển thành phân phối Gaussian (chuẩn tắc) từ đó giá trị trung bình 0 và độ lệch chuẩn là 1
                - Nếu dữ liệu bên trong không được phân phối chuẩn thì đây không phải là cách chia tỷ lệ tốt nhất để sử dụng
                - Khi nào thì sử dụng
                    * - Khi các thuộc tính số mà chúng ta đưa vào đều là phân phối chuẩn hoặc xấp xỉ chuẩn 
                    => hiệu quả cao với standardScaler phải sử dụng khi thuộc tính phân phối chuẩn hoặc xấp xỉ chuẩn 
                    => Xem hiệu quả cao của model hay không.
            3.4.2. MinMaxScaler
                - MinMaxScaler có thể được xem là thuật toán chia tỉ lệ nổi tiếng nhất và tuân theo công thức sau cho mỗi tính năng/ thuộc tính
                - Công thức: (xi - min(x))/(max(x) - min(x)) => Dùng để thu hẹp dữ liệu lại min =1 và max = 1
                - Về cơ bản, nó thu hẹp phạm vì sao cho phạm vi mới nằm trong khoảng từ 0 đến 1 (hoặc -1 đến 1 nếu có các giá trị âm)
                - Phương pháp được để xuất sử dụng trong thuộc tính của chung ta có 1 hay vài thuộc tính không có phân phối chuẩn hoặc xấp xỉ không chuẩn
                - Đặc điểm
                    * Bộ chia tỉ lệ này hoạt động tốt hơn trong các trường hợp mà bộ chia tỷ lệ tiêu chuẩn (standard scaler) có thể không hoạt động tốt
                    * Nếu phân phối không phải là Gaussian hoặc độ lệch chuẩn là rất nhỏ, min-max scaler hoạt động tốt hơn
                    * Động lực để sử dụng tỉ lệ này bao gồm độ lệch chuẩn của các tính năng rất nhỏ và duy trì các entry có giá trị 0 trong dữ liệu thưa thớt
                - Khi nào nên dùng
                    * Không nên dùng: Có outlier không hợp lệ
                    * Không có outlier là điều kiên quyết => Ảnh hưởng tỉ lệ scale
                    * Không có phân phối chuẩn hoặc không phân phối xấp xỉ không chuẩn
                    * Nếu là outlier hợp lệ thì sau khi giải quyết xong outlier thì chung ta có quyền sử dụng MinMaxOutlier
                - Những hệ số tương quan là không thay đổi: Knew() => Phân phối chuẩn không thay đổi
                - Ưu tiền dùng khi nào
                    * Thuộc tính input không có phân phối chuẩn hoặc xấp xỉ không chuẩn
                    * Dữ liệu của chung ta không có outlier
                    => Khi có 2 yếu tố đó thì mới sử dụng MinMaxOutlier       
            3.4.3. Robust Scaler
                - MinMaxScaler dùng toàn bộ dữ liệu scaler tính toán ra giá trị/ tỉ lệ scaler
                - Robust Scaler dùng phần dữ liệu năm ở vị Q1 và Q3, nằm trong dữ liệu năm trong khoảng IQR để tính toán toàn bộ scaler
                - RobustScaler sử dụng một phương pháp tương tự như MinMaxScaler nhưng nó sử dụng Interquartike range thay cho min - max
                - Công thức: (xi-Q1(x))/(Q3(x)-Q1(x))
                - Đặc điểm
                    * Điều này có nghĩa là RobustScaler đang sử dụng ít dữ liệu hơn khi chia tỉ lệ để nó phù hợp hơn khi có cac ngoại lệ trong dữ liệu
                    * Sau khi áp dụng Robust scaling, các phân phối được dựa vào cùng một tỷ lệ và trùng lặp, 
                    nhưng các ngoại lệ vẫn nằm ngoài phần lớn của các bản phân phối mới
            3.4.4. Normalizer
                - Dùng cho cho dữ liệu rời rạc
                - Dùng cho 2 thuật toán: KNN và Neuron network
            3.4.5. Binarizer
                - Chuyển đổi dữ liệu bằng cách dùng binary threshold (Ngưỡng chọn).
                Tất cả các giá trị trên threshold được thay bằng 1 và các giá trị <= threshold được thay thế bằng 0
    4. Tóm tắt các phương pháp
        (1) StandardScaler: Khi thuộc tính đều có phân phối chuẩn và xấp xỉ phân phối chuẩn 
        (2) MinMaxScaler: Không có phân phối chuẩn và xấp xỉ phân phối chuẩn và trong dữ liệu không có outlier không hợp lệ
        (3) RobustScaler: Không có phân phối chuẩn và xấp xỉ phân phối chuẩn và trong dữ liệu có outlier
        (4) Có thể sử dụng log normalization để thay đổi phân phối chuẩn lệch phải để rồi sử dụng StandardScaler
    '''

import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
warnings.filterwarnings('ignore')

## A. LOG NORMALIZATION
### 2.1. Chuẩn hoá Log normalization
##### lấy log cho các thuộc tính intputs liên tụ => Cách áp dụng phân phối chuẩn lệch phải thì cải thiện thuộc rất tốt và áp dụng cho 1 thuộc tính
'''

'''
def log_normalization(lst_lientuc_chosen, lst_lientuc_chosen2, df):
    try:
        for i in lst_lientuc_chosen:
            if i in lst_lientuc_chosen2: 
                pass
            else:
                name_log = i + '_log'
                lst_lientuc_chosen2.append(name_log)
                df[name_log] = np.log(df[i])
            
        result = df[lst_lientuc_chosen2]
        return result
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 2.2. Trực quan hóa thuộc tính trước log normalization
def visualization_before_log_normalization(df, var_before_log):
    result = (
        plt.subplot(1,2,1),
        plt.hist(df[var_before_log]),
        plt.subplot(1,2,2),
        sns.displot(df[var_before_log]),
        plt.show()
        )
    return result

### 2.2. Trực quan hóa thuộc tính sau log normalization
def visualization_after_log_normalization(df, var_after_log):
    result = (    
        plt.subplot(1,2,1),
        plt.hist(df[var_after_log]),
        plt.subplot(1,2,2),
        sns.displot(df[var_after_log]),
        plt.show())
    return result

### 2.3. Xem mối quan hệ 2 biến trước log
def visualization_before_log_normalization_relationship(df, var_befor_log_x, var_before_log_y):
    result = (
        sns.lmplot(data=df, x = var_befor_log_x, y=var_before_log_y)
    )
    return result
    
### 2.4. Xem mối quan hệ 2 biến sau log
def visualization_after_log_normalization_relationship(df, var_after_log_x, var_after_log_y):
    result = (
        sns.lmplot(data=df, x = var_after_log_x, y=var_after_log_y)
    )
    return result

## B. Feature scalling
## 3.1 Trực quan hóa dữ liệu và xem dữ liệu
### 3.1.1. Trực quan hóa dữ liệu thuộc tính
def visualization_feature_hist_subplot(df, var_featuer):
    result = (
        plt.subplot(1,2,1),
        plt.hist(df[var_featuer]),
        plt.subplot(1,2,2),
        sns.displot(df[var_featuer]),
        plt.show()
    )
    return result

### 3.1.2. Xem phân phối chuẩn lệch trái hay phải
def skew_feature(df, var_featue):
    result = df[var_featue].skew()
    return result

### 3.1.2. Xem phân phối chuẩn nhọn hay bẹt
def kurtosis_feature(df, var_featuer):
    result = df[var_featuer].kurtosis()
    return result

### 3.1.4. Trực quan hóa dữ liệu outlier
def visualization_feature_boxplot(df, var_featuer):
    result = (plt.boxplot(df[var_featuer])
                ,plt.show())
    return result
### 3.1.5. feature scaler visualization
def visualization_robust_scaler(df, var_before_scaler1, var_before_scaler2, var_after_scaler1, var_after_scaler2):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5)),
    result = (
        ax1.set_title("Before Sacling"),
        sns.kdeplot(df[var_before_scaler1], ax=ax1),
        sns.kdeplot(df[var_before_scaler2], ax=ax1),

        ax2.set_title("After Robust Sacler"),
        sns.kdeplot(df[var_after_scaler1], ax=ax2),
        sns.kdeplot(df[var_after_scaler2], ax=ax2),
        plt.show()
    )
    return result

### 3.2. Robust scaler: Phân phổi không chuẩn hoặc không xấp xỉ chuẩn; và có outlier
### 3.2.1. Robust scaler
##### Trả về dữ liệu là 1 dataframe sau đó dùng concat để đưa vào df
def robust_Scaler(df, lst_lientuc_chosen):
    try:
        # Thêm tên khác cho các thuộc tinh scaler
        lst_name_column = []
        for i in lst_lientuc_chosen:
            lst_name_column.append(i+'_scaler')
        # Chuẩn hoá bằng RobustScaler trên dữ liệu đã Log normalization
        #--> Do các thuộc tính không có PP chuẩn và có outliers nên không sử dụng StandarScaler/MinMaxScaler
        scaler = RobustScaler()
        data = df[lst_lientuc_chosen]
        # X_train_scale = scaler.fit_transform(X_before_scale)
        scaler = scaler.fit(data)
        df_new = scaler.transform(data)
        df_new = pd.DataFrame(df_new, columns=lst_name_column)
        return df_new
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.3. Standard scaler: Nhớ chuẩn hóa dữ liệu trước khi làm
### 3.3.1. Standard scaler
### Trả về 1 dataframe
def min_max_scaler(df, lst_lientuc_chosen):
    try:
        # Thêm tên khác cho các thuộc tinh scaler
        lst_name_column = []
        for i in lst_lientuc_chosen:
            lst_name_column.append(i+'_scaler')

        # Chuẩn hoá bằng RobustScaler trên dữ liệu đã Log normalization
        #--> Do các thuộc tính không có PP chuẩn và có outliers nên không sử dụng StandarScaler/MinMaxScaler
        scaler = StandardScaler()
        data = df[lst_lientuc_chosen]
        data = data.astype('float64')
        # X_train_scale = scaler.fit_transform(X_before_scale)
        scaler = scaler.fit(data)
        df_new = scaler.transform(data)

        df_new = pd.DataFrame(df_new, columns=lst_name_column)
        return df_new
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

### 3.3. Standard scaler: Nhớ chuẩn hóa dữ liệu trước khi làm
### 3.3.1. Standard scaler
### Trả về 1 dataframe
def standard_scaler(df, lst_lientuc_chosen):
    try:
        # Thêm tên khác cho các thuộc tinh scaler
        lst_name_column = []
        for i in lst_lientuc_chosen:
            lst_name_column.append(i+'_scaler')

        # Chuẩn hoá bằng RobustScaler trên dữ liệu đã Log normalization
        #--> Do các thuộc tính không có PP chuẩn và có outliers nên không sử dụng StandarScaler/MinMaxScaler
        scaler = MinMaxScaler()
        data = df[lst_lientuc_chosen]
        # data = data.astype('float64')
        # X_train_scale = scaler.fit_transform(X_before_scale)
        scaler = scaler.fit(data)
        df_new = scaler.transform(data)

        df_new = pd.DataFrame(df_new, columns=lst_name_column)
        return df_new
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

###################################################################################################################################################################
#                                               FEATURE ENGINEERING - KỸ THUẬT CHUYỂN ĐỔI NHÃN CHO THUỘC TÍNH                                                     #
###################################################################################################################################################################
Feature_Engineering_KY_THUAT_CHUYEN_DOI_NHAN ='''
    A. Xử lý categorical và Continiuos
        1. Giới thiếu chung
            - Feature engineering: dùng domain business để tạo ra các tính năng giúp thuật toán máy học hoạt động
            - Cung cấp 1 số thư vện để chuyển đổi kiểu dữ liệu categorical từ dạng object sang kiểu dữ số
            - Giúp cho chung ta có thể truy vết nếu có vấn đề
            - Vậy vì sao phải chuyển qua kiểu category từ text sang dạng số ?
                * Tiết kiệm bộ nhớ
                * Các thuật toán Machine learning sẽ thao tác một cách nhanh hơn kiểu dữ liệu object
                * Bản chất thì chuyển qua kiểu dữ liệu category nhưng bản chất bên trong là kiểu object nên phải chuyển qua kiểu số để thuật toán ML thực hiện
        2. Đặc điểm
            - Feature engineering là việc tạo các tính năng đầu vào mới từ những tính năng hiện có của bộ dữ liệu
            - Đây là một trong những nhiệm vụ có giá trị nhất ma một nhà khoa học dữ liệu có thể làm để cải thiện hiệu suất mô hình
                * Chúng ta có thể làm cô lập và làm nổi bật thông tin chính, giúp thuật toán tập trung vào nhưng gì quan trọng
                * Chúng ta có thể vận dụng domain knowledge để có tính năng thích hợp
                * Chúng ta cũng có thể đem đến những người khác kiến thức mới và domain knowledge mới
        3. Domain knowledge
            - Chúng ta có thể thiết kế các thuộc tính thông tin (informative feature) 
            bằng cách dựa trên kiến thức chuyên môn của mình (hoặc những người khác) về lĩnh vực làm việc
            - Với những thông tin cụ thể mà ta có thể muốn cô lập, ta có thế có rất nhiêu tự do và sáng tạo
            - Domain knowledge rất rộng mở. Khi đó chũng ta có thể cạn kiệt về ý tưởng. Và có 1 số phương pháp phỏng đoán cụ thể giúp gợi mở nhiều hơn
        4. Tạo thuộc tính
            - Trong phần này chúng ta sẽ tìm hiểu các kỹ thuật liên quan đến thuộc tính tính năng (feature engineering) 
            và cách áp dụng nó vào dữ liệu trong thế giới thực
            - Chung ta sẽ tải, khám phá và trực quan hóa bộ dữ liệu, tìm hiểu về các loại dữ liệu cơ bản của chúng
            và lý do tại sao chúng có ảnh hưởng đến cách chúng ta thiết kế các thuộc tính của 
            a. Các kiểu dữ liệu khác nhau
                - Continuous: dữ liệu liên tục có thể là integer hoặc float
                - Categorical: dữ liệu phân loại, là tập hợp giá trị giới hạn: gender, country of birth
                - Ordinal: Giá trị xếp hạng, thường không có chi tiết về khoảng cách giữa chúng
                - Boolean: giá trị True/False
                - Datetime: giá trị thời gian
            b. Các thuộc tính thông dụng
                - df.columns: xem danh sách tên cột
                - df.dtypes: xem danh sách kiểu dữ liệu của cột
                - Chọn các cột có kiệu dữ liệu cụ thể: df2 = df.select_dtypes(include = ['int','float']) => df2.columns
        5. Chuyển đội dữ liệu
            - Integer encoder/ label encoder
                * Áp dụng trong kiểu phân loại và theo thứ tự
            - One hot encoder/ Dummy encoder
                * Đối với các biến phân loại không tồn tại mối quan hệ thứ tự, integer encoder là không đủ
                * Tuy thuộc vào dữ liệu chúng ta có, chung ta có thể gặp phải tình huống, sau khi Laber Encoder, 
                có thể nhầm lẫn mô hình vì nghĩ rằng một cột có dữ liệu với thứ tự hoặc thứ bậc nào đó.
                * One-hot encoder/ Dummy encoder có thể áp dụng thay cho biểu diễn số nguyên => kết quả giống nhau cách thức thực hiện khác nhau
            - One hot encoder      
                Bước 1: Xác định thuộc tính phân loại và không có thứ tự
                Bước 2: Import thêm thư viện vào
                Bước 3: Khởi tạo thư viện onehotencoder()
                Bước 4: Biển thuộc tính cần biến đổi theo mảng 2 chiều
                Bước 5: Truy ngược lại dữ liệu
            - Dummy encoder
                - Xử lý đơn giản hơn onehot encoder
                - Không lưu trữ dummy model encoder
                - Kết quả trả về là các cột mới trong dataFrame
                - Xử lý đơn giản đó là 1 phương thức của pandas không phải là model nên không thể truy ngược lại được
        6. Mục đích binning value
            - Đối với nhiều giá trị liên tục chúng ta có thể sẽ quan tâm ít hơn về giá trị chính xác của một cột kiểu số, thay vào đó chung ta quan tâm
            đến nhóm mà nó rơi vào => Điều này hữu ích khi vẽ các giá trị hoặc đơn giản hóa các mô hình ML Nó chủ yếu được sử dụng trên các biến liên tục 
            trong đó dộ chính xác không phải mối quan tâm lớn nhất. Tạo khoảng tuổi, chiều cao, tiền lương
            - Động lực chính của việc tao bin là làm cho mô hình mạnh mẽ hơn và ngăn ngừa overfitting, tuy nhiên, nó có chi phí hiệu suất cao hơn
            => Mỗi khi chúng ta dùng bin thì phải chấp nhận giảm bớt thông tin cho trường đó
            - Sự đánh đổi giữa hiệu suất và overfitting là điểm mẫu chốt của quá trình tạo bin
            => Nhìn chung chung ta có các cột số, ngoại trừ một số trường hợp quá rõ ràng, việc tạo bin có thể là dư thừa đối với một số loại thuật toán
            do ảnh hưởng của nó đến hiệu suất mô hình
            - Đối với các cột phân loại các nhãn có tần số thấp có thể ảnh hưởng tiêu cực đến độ mạnh của các mô hình thống kê.
            => Do đó việc gán danh mục chung cho các giá trị ít thường xuyên này sẽ giúp duy trì sức mạnh mẽ của mô hình
            - Các bin được tạo ra bằng cách sử dụng pd.cut(df['column_name'], bins) => Trong đó bins là integer chỉ định số lượng bin cách đều nhau hoặc đó 
            danh giới của bin
        7. Xử lý các danh mục không phổ biến (uncommon category)
            Một số tính năng có thể có nhiều loại khác nhau những phân phối rất không đồng đều về sự xuất hiện của chúng
    B. Xử lý dữ liệu văn bản (Text data)
        1. Các xử lý văn bản - text data
            - Dữ liệu văn bản phi cấu trúc có thể được sử dụng trực tiếp trong hầu hết các phân tích
                Bước 1: Chuẩn hóa dữ liệu và loại bỏ các ký tự nào có thể gây ra sự cố sau này trong việc phân tích của bản
                    - Loại bỏ những ký tự không mong muốn 
                        * Dùng regular expression:
                            + [a-zA-Z]: Tất cả các ký tự chữ
                            + [^a-zA-Z]: tất cả các ký tự không phải là ký tự chữ
                    - Chuẩn hóa chữ: dùng str.lower(): chuyển sang chữa thường
                Bước 2: Thuộc tính văn bản cấp cao (High level text feature)
                    - Khi văn bản đã được làm sạch và chuẩn hóa, chung ta có thể bắt đầu tạo các tính năng từ dữ liệu.
                    => Dùng các tính năng độ dài và số lượng tử
                        * Dùng str.len() biết chiều dài của chuỗi
                        * Dùng str.split() để cắt chuỗi thành các phần tử chữa trong list
                Bước 3: Word count Representation
                    - Khi thông tin cấp cao đã được ghi lại, chúng ta có thể bắt đầu tạo các thuộc tính dữ trên nội dung thực tế của từng văn bản
                    - Một cách để làm là tiếp cận nó theo cách tương tự như cách đã làm việc với các biến phân loại
                        * Đối với mỗi từ duy nhất trong tập dữ liệu tạo ra một cột
                        * Đối với entry, số lần từ này xuất hiện được đếm và giá trị đếm được nhập vào cột tương ứng
                    - Các cột count này có thể được sử dụng để huấn luyện các mô hình machine learning
                    - Kỹ thuật này được gọi là bag of words
        2. Dùng CountVectorizer
            - Các bước thực hiện trong thư viện
                Bước 1: Khởi tọa CountVectorizer
                Bước 2: fit
                Bước 3: Chuyển đổi văn bản
                Bước 4: Chuyển đổi kết quả các cột trong dataframe
                Note:
                    - CountVectorizer bổ sung tham số min_df và max_df
                    - Như vậy dùng CountVectorizer mặc định sẽ tạo ra một tính năng cho mỗi từ đơn lẻ trong kho văn bản
                    => Nó tạo ra nhiều thuộc tính, bao gồm các giá trị cũng cấp rất ít giá trị phân tích
                    - Cho nên min_df và max_df dùng để giảm dố lượng thuộc tính không cần thiết:
                        * min_df: Chỉ sử dụng những từ xuất hiền nhiều hơn tỷ lệ phần trăm tài liệu => Loại bỏ những từ ít hơn không khái quán được văn bản
                        * max_df: Chỉ sử dụng các từ xuất hiện ít hơn tỷ lệ phần % tài liệu này => Việc này làm giảm bớt những từ phổ biến xãy ra trong văn bản
                        mà không có thêm giá trị "and" hoặc "the"
                        VD: min_df > 20% và max_df < 80%
                Bước 5: Gộp dataFrame
                    df_new = pd.concat([df1, df2], axis=1, sort=False)
        3. Dùng TF-IDF
            - Tf-Idf (Term frequency-inverse document frequency)
                * Xử lý ngôn ngữ tự nhiên là một kĩ thuật quan trọng nhằm giúp máy tính hiểu được ngôn ngữ của con người,
                qua đó hướng dẫn máy tính thực hiện và giúp đỡ con người trong những công việc có liên quan đến ngôn ngữ như:
                dịch thuật, phân tích dữ liệu văn bản, nhân dạng tiếng nói, tìm kiếm thông tin, tóm tắt vắn bản...
                * Một trong những kĩ thuật để xử lý ngôn ngữ tự nhiên là TF-IDF (Tần xuất xuất hiện của từ nghịch đảo tần xuất suất của văn bản)
                * Mặc dù số lần xuất hiện của các từ có thể có ích khi xây dựng các mô hình, các từ xuất hiện nhiều lần có thể làm sai lệch kết quản
                một cách không mong muốn => TF-IDF có tác dụng làm giảm giá trị của các từ phổ biến, đồng thời tăng trọng số của các từ không xảy
                ra trong nhiều tài liêu
                * TF-IDF: là trọng số của một từ trong văn bản thu được qua thống kê, nó thể hiện mức độ quan trọng của từ trong một văn bản, 
                với bản thân văn bản đang xét nằm trong một tập hợp các văn bản
                * IF-IDF thường được sử dụng vì: trong ngôn ngữ luôn có những từ xảy ra thường xuyên với các từ khác
'''
import pandas as pd 
import warnings
import pandas_profiling as pp # tổng quan ban đầu về dữ liệu => Cài trên này
warnings.filterwarnings('ignore')
import re
import string
import langid # Dùng để xem ngôn ngữ nước nào
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# sử dụng thư viện CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# A. FEATURE ENGINEERING
## 1.1. Get dummy: dùng để gán nhãn cho thuộc tính của thư viện pandas
def get_dummies(df, feature_get_dum, prefix):
    df_new = pd.get_dummies(data=df, columns=[feature_get_dum], prefix=prefix)
    return df_new

## 1.2. Label_Encoder: Áp dụng trong kiểu phân loại và theo thứ tự
def label_Encoder(df, name_before_label_encoder, name_after_label_encoder):
    encoder = LabelEncoder()
    df[name_after_label_encoder] = encoder.fit_transform(df[name_before_label_encoder])
    return df

## 1.3. MixMaxScaler: Có outlier và Không có phân phối chuẩn hoặc lệch chuẩn 
### Biến đổi bằng Dummy Encoder/OneHotEncoder
def onehotencoder(df, lst_phanloai_chosen):
    encoder = OneHotEncoder()
    lst_encode = lst_phanloai_chosen
    # lst_encode.remove('symboling')
    arr = encoder.fit_transform(df[lst_encode]).toarray()
    cols = []
    n = 0
    for i in encoder.categories_:
        for j in i[1:]: 
            t = 'oh_' + lst_encode[n] + '_' +j
            t = t.replace('-', '_')
            cols.append(t)
        n = n+1

    X_oh_encode = pd.DataFrame(arr, columns=cols)
    return X_oh_encode

## 1.4. Binning Value: Rất quan trong dùng để xem mức độ chia dữ liệu thế nào trong dữ liệu continious
def binning_value(df, name_before_binning_value, name_after_binning_value, bins ):
    df[name_after_binning_value] = pd.cut(df[name_before_binning_value], bins=bins) 
    return df


## B. XỬ LÝ DỮ LIỆU TEXT TIẾNG ANH
### 2.1. Xử lý dữ liệu text
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

### 2.2. Check ngôn ngữ tiếng anh
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

### 2.3. Remove features
##### Remove những dữ liệu có chứa những ký tự đặc biệt và các ký tự không cần thiết
def removeFeatures(dataStr):
    try:
        # compile regex
        url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
        punc_re = re.compile('[%s]' % re.escape(string.punctuation))
        num_re = re.compile('(\\d+)')
        mention_re = re.compile('@(\w+)')
        alpha_num_re = re.compile("[^A-Za-z0-9]") # Kiểu phủ định khác các chữ cái trên tron ^ là kiểu phủ đỉnh
        # convert to lowercase
        dataStr = dataStr.lower()
        # remove hyperlinks
        dataStr = url_re.sub(' ', dataStr)
        # remove @mentions
        dataStr = mention_re.sub(' ', dataStr)
        # remove puncuation
        dataStr = punc_re.sub(' ', dataStr)
        # remove numeric 'words'
        dataStr = num_re.sub(' ', dataStr)
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

### 2.4. removes stop words
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

### 2.5. Tagging text - Gắn thẻ vẵn bản
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

### 2.6. lemmatization
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

### 2.7. CountVectorizer: dùng để phân chia data text thành các chữa
def countVectorizer(df, var_feature_CountVectorizer,min_df=0.2, max_df=0.8):
    # khởi tạo
    cv = CountVectorizer(min_df=min_df, max_df=max_df)
    cv_transformed = cv.fit_transform(df[var_feature_CountVectorizer])
    cv_transformed = cv_transformed.toarray()
    # print(cv.get_feature_names())
    cv_df_new = pd.DataFrame(cv_transformed, columns=cv.get_feature_names()).add_prefix('cv_')
    return cv_df_new

### 2.8. TF-IDF
def tf_Idf(df, var_feature_CountVectorizer, max_features=200, stop_words='english'):
    tf_idf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tf_transformed = tf_idf.fit_transform(df[var_feature_CountVectorizer])
    tf_transformed = tf_transformed.toarray()
    tf_df = pd.DataFrame(tf_transformed, columns=tf_idf.get_feature_names()).add_prefix('tf_')
    return tf_df

### 2.9. TF_IDF với ngram_range
def tf_Idf_ngram_range(df, var_feature_CountVectorizer, max_features=200, stop_words='english',ngram_range=(2,2)  ):
    tf_idf = TfidfVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=ngram_range)
    tf_transformed = tf_idf.fit_transform(df[var_feature_CountVectorizer])
    #print(tf_idf.get_feature_names())
    tf_transformed = tf_transformed.toarray()
    tf_df = pd.DataFrame(tf_transformed, columns=tf_idf.get_feature_names()).add_prefix('tf2_')
    return tf_df



####################################################################################################################################################################
#                                                                       DATA ANALYSIS                                                                              #
####################################################################################################################################################################
## SUMMARY DATA ANALYSIS - TỔNG QUAN VỀ PHÂN TÍCH DỮ LIỆU
data_analysis = '''
    A. Quy trình Data Analysis
        1. Định nghĩa
            - Data analysis - phân tích dữ liệu là quá trình kiểm tra, làm sach, 
            chuyển đổi và mô hình hóa dữ liệu với mục tiêu khám phá thông tin hữu ích, thông báo và hỗ trợ ra quyết định.
            - Sử dụng trong nhiều lĩnh vực kinh doanh, khoa học và khoa học xã hội
            - Trong kinh doanh, phân tích dữ liệu đóng vai trò giúp đưa ra quyết định khoa học hơn và giúp doanh nghiệp hoạt động hiệu quả hơn
        2. Mục tiêu chính của Data analysis - Phân tích dữ liệu
            - Để có được thông tin hành động liên quan đến công việc/Doanh nghiệp...
            - Đối với nhà quản trị thì làm việc trên miền dữ liệu, đễ hỗ trợ các quyết định và hành động kinh doanh
            - Cho nên việc hiểu các lý thuyết và kỹ thuật trong các lĩnh vực phân tích dữ liệu như thống kê, khai thác dữ liệu và phân tích dự đoán 
            (statistics, data mining và predictive analytics) cực kỳ quan trọng.
        3. Một số kỹ thuật phân tích dữ liệu
            Phần 1: Phân tích tương quan - Correlation Analysis
                - Correlation là một kỹ thuật thống kê có thể cho thấy mối quan hệ giữa 2 biến
                - Xác minh và làm rõ mối liên hệ giữa 2 thuộc tính với nhau có mối quan hệ, ràng buộc, chi phối.... đọc lập hay không độc lập
            Phần 2: Phân tích hồi quy - Regression Analysis
                - Đây là một trong những kỹ thuật phân tích dữ liệu thống kê điều tra mối quan hệ giữa các biến khác nhau
                - Được sử dụng khi nghi ngờ rằng một trong các biến có thể ảnh hưởng đến (các) biến khác
                - Phân tích hồi quy có thể được sử dụng khi cố gắng tạo dự báo hoặc phân tích mối quan hệ giữa các biến
            Phần 3: Trực quan hóa dữ liệu - Data visualization
                - Trực quan hóa dữ liệu giúp mọi người hiểu được thông tin quan trọng trong dữ liệu bằng cách hiển thị nó trong bối cảnh trực quan
                - Một trong những kỹ thuật phân tích quan trọng nhất hiện nay khi thế giới đầy dữ liệu
                - Đặc biệt hữu ích khi chúng ta tìm cách năm bắt những hiểu biết sâu sắc từ một khối lượng lớn dữ liệu một cách nhanh chóng
                - Có thể dùng các tool: power BI, Google studio, tablue...
            Phần 4: Phân tích tình huống/ Kịch bản - Scenario Analysis
                - Phân tích tình huống/ kịch bản là phân tích các sự kiện có thể xảy ra trong tương lai với kết quả thay thế
                - Được sử dụng khi chung ta có 1 số lựa chọn thay thế tiềm năng nhưng không chắc chắn về quyết định đưa ra
                - Phân tích kịch bản là một công cụ quan trọng trong nhiều DN và được sử dụng rộng rãi để đưa ra dự đoán cho tương lai
            Phần 5: Khai thác dữ liệu - Data Mining
                - Khai thác dữ liệu đôi khi là khám phá dữ liệu/ kiến thức là 1 quá trình phân tích dữ liệu được thiết kế được làm việc với khối 
                lượng dữ liệu lớn giúp phát triển các mẫu, mỗi quan hệ hoặc thông tin có liên quan có thể cải thiện hiệu suất
                - Ví dụ: Nhà bán lẽ có thể sử dụng khai thác dữ liệu để công ty thẻ tín dụng để xuất sản phẩm cho chủ thẻ dựa trên phân tích chi 
                tiêu hàng tháng của họ.
    4. Kỹ thuật nâng cao
            Phần 1: Mạng neuron - Neural Networks
                - Mạng neuron là kỹ thuật lấy cảm hứng từ cách mạng thần kinh sinh học, như não, để xử lý thông tin
                - Mục đích của các mạng này là mô phỏng quá trình học tập của bộ não con người trên máy tính để tạo điều kiện cho việc 
                ra quyết định trong trí tuệ nhân tạo (AI)
                - Kỹ thuật Neural Networks có thể được sử dụng để trích xuất các mẫu và 
                phát hiện các xu hướng quá phức tạp để được xác định bởi con người hoặc các kỹ thuật máy tính khác.
                - Mạng lưới thần kinh được đào tạo có thể xem như 1 chuyên gia có khả năng đưa ra các dự đoán với 
                các tình huống đã cho và trả lời câu hỏi "what if"
                - Khả năng tự học hỏi dựa trên kỹ thuật này
            Phần 2: A/B testing
                - A/B testing có thể gọi là thử nghiệm phân tách. 
                Đây là 1 phương pháp so sánh hai phiên bản của một trang web hoặc ứng dụng với nhau để xác định phiên bản nào hoạt động tốt hơn.
                - Kỹ thuật A/B testing thường được sử dụng trong tiếp thị kỹ thuật số để kiểm tra phản ứng của người dùng đồi 
                với 1 tin nhắn và xem cái nào hoạt động tốt nhất, kiểm tra giả thuyết trong việc ra mắt một hoạt động mới, 
                một chiến dịch quảng cảo hoặc 1 thông điệp quảng cáo
                - Phiên bản nào hoạt động tốt hơn thì làm biên bản chính để hoạt động của chúng ta
        5. Công cụ dành cho phân tích dữ liệu        
            Phần 1: Công cụ chính tạo nên phân tích dữ liệu là dòng và bảng
                - Tạo bảng phân phối tân suất (frequency distribution table) để hiện thị dữ liệu cột và dòng mỗi tương quan giữa 2 thuộc tính
                - Rất hiểu ít với 2 thuộc tính categorical
            Phần 2: Công chính tiếp theo đó là trực quan hóa dữ liệu
                - Dùng các tools để thực xự rất đẹp
        6. Ba quy tắc phân tích dữ liệu
            Quy tắc 1: Nhìn vào dữ liệu và suy nghĩ vào những gì chúng ta muốn biết ?
                - Đặt câu hỏi và đóng khung câu hỏi và xem các giả thuyết
                - Ví dụ: Chung ta có muốn chứng minh trái đất hình cầu ?
            Quy tắc 2: Ước tính 1 xu hướng trung tâm (central Tendency) cho dữ liệu
                - Ví dụ: mean, median
                - Cái nào chúng sử dụng sẽ phụ thuộc vào giả thiết của trong quy tắc 1
            Quy tắc 3: Xem xét các ngoại lệ cho xu hướng trung tâm
                - Nếu đã đo trung bình, hãy nhìn vào số liệu không phải trung bình.
                - Nếu đã đo được 1 trung vị, hãy nhìn vào những con số mà không đáp ứng được kỳ vọng đó
                - Ngoại lệ giúp bạn phát hiện vấn đề với kết luận
                - Nhưng nếu nhìn vào các ngoại lệ, bạn có thể thấy họ đang nhận được 100 trong 3 lớp và 40 trong ba lớp khác. 
                => Trong trường hợp này trung bình là hoàn toàn sai lệch.
        7. Vấn đề của phân tích dữ liệu
            - Tại sao nhiều trường hợp phân tích dữ liệu kết thúc với tuyên bố bị lỗi ? 
            Một trong những lý do chính là một quá trình phức tạp và tẻ nhạt. Nó không bao giờ dễ dàng như đưa số vào máy tính.
                * Tập dữ liệu không được xử lý sách sẽ và chuẩn hóa bên trong
                * Sử dụng sai phương pháp
                * Khả năng lỗi phát sinh trong quá trình làm việc rất là cao
                * Nhiệm vụ sử dụng công cụ nào cho hợp lý => Những biểu đồ quyết định phù hợp
            - Một số vấn đề có thể dẫn đến phân tích dữ liệu bị lỗi
                * Không có kỹ năng phân tích đúng
                * Sử dụng công cụ sai để phân tích dữ liệu. Ví dụ: sử dụng z-score khi dữ liệu không có phân phối chuẩn
                * Để bias ảnh hưởng đến kết quả => Bị chi phối bởi định kiến của bên ngoài và môi trường
                * Không tìm ra ý nghĩa thống kê => Không xác định được yêu cầu của bài toán và giả thiết không đúng nên đi lòng vòng
                * Phát biểu không chính xác null hypothesis và alternate hypothesis
                * Sử dụng graph và chart không chính xác gây hiểu lầm
        8. Quy trình Data analysis
            - Business understanding => Data Requirements => Data conllection => Data Pre-processing 
            => Exploratory Data => Modeling Algorithms => Data product => Communication
            - Quy trình chi tiết
                Step1: Business understanding
                    * Trước khi cố gắng rút ra thông tin chi tiết hữu ích từ dữ liệu, điều cần thiết là xác định vấn đề kinh doanh cần giải quyết, 
                    cố gắng hiểu rõ về những gì doanh nghiệp cần trích xuất từ dữ liệu	
                    * Xác định các vấn đề (problem denfinition) là động lực để thực hiện kế hoạch phân tích dữ liệu. Các nhiệm vụ là xác định mục tiêu 
                    của phân tích, xác định các công việc, vạch ra vai trò và trách nhiệm, thu thập trạng thái hiện tại của dữ liệu,
                    xác định thời gian biểu và thực hiện phân tích chi phí / lợi nhuận. Từ đó một kế hoạch thực thi có thể được tạo ra.
                Step2: Data Requirements
                    * Dữ liệu là cần thiết để làm đầu vào cho phân tích, được chỉ định dựa trên yêu cầu của những người chỉ đạo phân tích 
                    hoặc khách hàng (những người sẽ sử dụng sản phẩm của phân tích). Mẫu mà dữ liệu sẽ được thu thập được gọi là 
                    một đơn vị thủ nghiệm ( Ví dụ 1 người hay 1 quần thể)
                    * Các biến cụ thể liên quan đến người (ví dụ: tuổi và thu nhập) có thể được chỉ định và thu được. 
                    Dữ liệu có thể là numerical hoặc categorical
                Step3: Data Collection
                    * Dữ liệu được thu thập từ nhiều nguồn khác nhau
                    * Các yêu cầu có thể được các nhà phân tích truyền đạt tời người gián sát dữ liệu
                    * Dữ liệu cũng được thu thập từ các cảm biến trong môi trường, chẳng hạn như camera giao thông, vệ tinh, thiết bị..
                    * Nó cũng có thể lấy từ các cuộc phỏng vấn, tải xuống từ các nguồn trực tuyến hoặc tài liệu đọc
                    * Bám sát vào data requirement để thu thập dữ liệu cho đúng
                Step4: Data pre-processing
                    * Dữ liệu thu được ban đầu phải được xử lý hoặc tổ chức để phân tích
                    * Các thao tác trong Data pre-processing
                        1. Data cleaning: làm sạch dữ liệu loại bỏ những lỗi tiềm ẩn bên trong
                        2. Data normalization: Chuẩn hóa dữ liệu - scaler dữ liệu
                        3. Data transformation: Biến đổi dữ liệu, biến đổi các thuộc tính, các tính năng phù hợp hơn
                        4. Missing values imputation: xử lý những dữ liệu phù hợp thay thế sao hợp lý hoặc loại bỏ
                        5. Data integration: tích hợp dữ liệu, cách đọc từ nhiều file, nối từ nhiều file, tích hợp bộ thống nhất
                        6. Noise identification: xử lý những thuộc tính hoặc giá trị nhiểu nào không
                Step5: Explpratory Data Analysis => Rất quan trọng
                    * Một dữ liệu đã được làm sạch, nó có thể được phân tích
                    * Các nhà phân tích có thể áp dụng các kỹ thuật được gọi là phân tích dữ liệu thăm dò có thể dẫn đến việc làm sạch 
                    dữ liệu bổ sung hoặc yêu cầu bổ sung cho dữ liệu => Các hoạt động này có thể lặp đi lặp lại cho đến khi về với bản chất
                    * Thống kê mô tả, những trung bình và trung vị có thể được tạo để giúp hiểu dữ liệu
                    * Trực quan hóa dữ liệu cũng có thể được sử dụng để kiểm tra dữ liệu ở định dạng đồ họa,
                    để có được cái nhìn sâu sắc về các thông điệp trong dữ liệu
                    * Vừa phân tích và khám phá dữ liệu để khám phá xem có dữ liệu làm sạch bổ sung hoặc thêm dữ liệu bổ sung
                Step6: Modeling & Algorithms
                    * Các công thức hoặc các mô hình toán học được gọi là thuật toán có thể được áp dụng cho dữ liệu để xác định mối quan hệ giữa các biến,
                    chẳng hạn như tương quan hoặc quan hệ nhân quả
                    * Các mô hình có thể được phát triển đánh giá một biến cụ thể trong dữ liệu dựa trên các biến khác trong dữ liệu, 
                    với 1 số lỗi còn lại tùy thuộc vào độ chính xác của mô hình (Data = model + Error)
                Step7: Data product
                    * Sản phẩm dữ liệu là một ứng dụng máy tính nhận dữ liệu đầu vào và đầu ra, đưa chúng trở lại môi trường
                    * Nó dự trên mô hình hoặc thuật toán
                    * Một ví dụ là một ứng dụng phân tích dữ liệu về lịch sử mua hàng của khách hàng và 
                    khuyến nghị các giao dịch mua khác mà khách hàng có thể được hưởng
                Step8: Communication
                    * Sau khi dữ liệu được phân tích, nó có thể được báo cáo theo nhiều định dạng cho người dùng để hỗ trọ các yêu cầu của họ.
                    Người dùng có thể có phản hồi, dẫn đến việc phân tích bổ sung
                    * Phần lớn chu trình phân tích lặp đi lặp lại
                Chốt vấn đề
                    * Khi xác định cách truyền đạt kết quả, nhà phân tích có thể xem xét các kỹ thuật trực quan hóa dữ liệu để giúp truyền đạt thông điệp rõ ràng và 
                    hiệu quả đến khán giả
                    * Trực quan hóa dữ liệu sử dụng hiển thị thông tin (Như bảng và biểu đồ) để giúp truyền đạt các thông điệp chính cho trong dữ liệu
                    - Các bảng hữu ích cho người dùng có thể tra cứu các số cụ thể, trong các biểu đồ (barplot, line plot...) 
                    có thể giúp giải thích thông điệp định lượng có trong dữ liệu
    B. Exploratory Data Analysis (EDA) - Phân tích khám phá dữ liệu
        1. Giới thiếu
            - EDA: là 1 cách tiếp cận để phân tích dữ liệu => EDA nghiên cứu 1 cái nhìn về dữ liệu và cố gắng hiểu về dữ liệu
            - EDA: quá trình quan trọng trong việc thực hiện điều tra ban đầu về dữ liệu để khám phá các mẫu, phát hiện dị thường, kiểm tra giả thuyết và
            kiểm tra các giả định với sự trợ giúp của thống kê tóm tắt và biểu diễn đồ họa
            - EDA: liên quan đến việc nhà phân tích cố gắng để có được một cảm giác của người dùng cho bộ dữ liệu, thường sử dụng phán đoán của chính họ để
            xác định yếu tố quan trọng nhất trong bộ dữ liệu là gì
        2. Mục đích của EDA
            - Kiểm tra dữ liệu bị thiếu và các lỗi khác
            - Có được cái nhìn sâu sắc tối đa về tập dữ liệu và cấu trúc cơ bản của nó
            - Khám phá 1 mô hình tốt, một mô hình giải thích dữ liệu với số lượng biến dự đoán tối thiểu
            - Kiểm tra các giả định liên quan đến bất kỳ mô hình phù hợp hoặc kiểm tra giả định
            - Tạo ra các danh sách ngoại lệ hoặc dị thường khác
            - Tìm ước tính tham số và khoảng tin cậy liên quan hoặc sai số
            - Xác định các biến có ảnh hưởng nhất
            - Kiến thức cụ thể khác có thể có được thông qua EDA như tạo danh sách xếp hạng các yếu tố liên quan. Có thể không nhất thiết phải có tất cả các mục trên
            trong phân tích dữ liệu
            

'''

## for data
import numpy as np
import pandas as pd
## for plotting
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ppscore
## for machine learning
from sklearn import preprocessing, impute, utils, model_selection
import imblearn


#############################################################################################     
## 1.1. Check data type is continious or categorical
'''
Recognize whether a column is numerical or categorical.
:parameter
    :param df: dataframe - input data
    :param col: list - name of the column to analyze => chuỗi column
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    lsts_: dictionary type 
'''
def utils_recognize_type_lst(df, cols, max_cat=20):
    try: 
        categoricals = []
        continious = []
        lsts_ = {}
        for col in cols:
            if (df[col].dtype == "O") | (df[col].nunique() < max_cat):
                categoricals.append(col)
            else:
                continious.append(col)
        lsts_ = {'lst_categoricals': categoricals, 'lst_continious': continious}
        return lsts_
    except Exception as failGeneral:
        print("Fail system, please call developer...", type(failGeneral).__name__)
        print("Mô tả:", failGeneral)
    finally:
        print("close")

## 1.2.1. Check data type: cat or num
'''
Recognize whether a column is numerical or categorical.
:parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical and "num" otherwise
'''
def utils_recognize_type_cat_or_num(df, col, max_cat=20):
    if (df[col].dtype == "O") | (df[col].nunique() < max_cat):
        return "categorical"
    else:
        return "Continious"

### 1.2.2. Check data type: cat or num
'''
Recognize whether a column is numerical or categorical.
:parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical and "num" otherwise
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"

## 1.3. Check all general overview of a dataframe
'''
Get a general overview of a dataframe.
:parameter
    :param dtf: dataframe - input data
    :param max_cat: num - mininum number of recognize column type
    : Choose dùng để chọn trực quan hóa theo cột => dataframe hoặc là visualization
    return dataframe or visualization
'''
def df_overview(df, choose= 'dataframe', max_cat=20, figsize=(10,5),  columns = ['Feature', 'Type_Feature', 'NAS', 'Check_data', "Check"]):
    ## recognize column type
    dic_cols = {col:utils_recognize_type_cat_or_num(df, col, max_cat=max_cat) for col in df.columns}
    ## print info
    len_dtf = len(df)
    lsts_ = []
    if choose == "dataframe":
        print("Shape:", df.shape)
        print("-----------------")
        for col in df.columns:
            info = col+" | "+dic_cols[col]
            info = info+" | Nas: "+str(df[col].isna().sum())+"("+str(int(df[col].isna().mean()*100))+"%)"
            if dic_cols[col] == "categorical":
                info = info+" | Categories: "+str(df[col].nunique())
            else:
                info = info+" | Min-Max: "+"({x})-({y})".format(x=str(int(df[col].min())), y=str(int(df[col].max())))
            if df[col].nunique() == len_dtf:
                info = info+" | Possible check"
            # print(info)
            x = info.split(" | ")
            lsts_.append(x)
        result = pd.DataFrame (lsts_, columns=columns)
        return result
    elif choose == "visualization":
         ## add legend
        print("\033[1;37;40m Categerocial \033[m", "\033[1;30;41m Numerical \033[m", "\033[1;30;47m NaN \033[m")
        ## plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = df.isnull()
        for k,v in dic_cols.items():
            if v == "Continious":
                heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
            else:
                heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
        result = (
            sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview'),
            #plt.setp(plt.xticks()[1], rotation=0)
            plt.show()
        )
        return result

## 1.4. Check the primary key of a df
'''
Check the primary key of a dtf
:parameter
    :param dtf: dataframe - input data
    :param pk: str - column name
'''
def check_pk(df, pk):
    unique_pk, len_dtf = df[pk].nunique(), len(df)
    check = "unique "+pk+": "+str(unique_pk)+"  |  len dtf: "+str(len_dtf)
    if unique_pk == len_dtf:
        msg = "OK!!!  "+check
        print(msg)
    else:
        msg = "WARNING!!!  "+check
        ERROR = df.groupby(pk).size().reset_index(name="count").sort_values(by="count", ascending=False)
        print(msg)
        print("Example: ", pk, "==", ERROR.iloc[0,0])

## 1.5. Moves columns into a df => front to end
'''
Moves columns into a df.
:parameter
    :param dtf: dataframe - input data
    :param lst_cols: list - names of the columns that must be moved
    :param where: str - "front" or "end"
:return
    df with moved columns
'''
def pop_columns(df, lst_cols, where="front"):
    current_cols = df.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        df = df[lst_cols + current_cols]
    elif where == "end":
        df = df[current_cols + lst_cols]
    return df

## 1.6. Plots the frequency distribution of a dtf column. => Check dữ liệu của dữ liệu categorical và continious
'''
Plots the frequency distribution of a dtf column. => Cho biết mức độ dữ liệu lệch trái hay phải
:parameter
    :param dtf: dataframe - input data
    :param x: str - column name
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
    :param top: num - plot setting
    :param show_perc: logic - plot setting
    :param bins: num - plot setting
    :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
    :param box_logscale: logic
    :param figsize: tuple - plot settings
'''
def freqdist_plot(dtf, x, max_cat=20, top=None, show_perc=True, bins=100, quantile_breaks=(0,10), box_logscale=False, figsize=(10,5)):
    try:
        ## cat --> freq
        if utils_recognize_type_cat_or_num(dtf, x, max_cat) == "categorical":   
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
            ax.grid(axis="x")
            result = (
                plt.suptitle(x, fontsize=20),
                plt.show()
            )
            return result
            
        ## num --> density
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade":True}, ax=ax[0])
            des = dtf[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            ### boxplot 
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_dtf = pd.DataFrame(dtf[x])
                tmp_dtf[x] = np.log(tmp_dtf[x])
                tmp_dtf.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                dtf.boxplot(column=x, ax=ax[1])
            result = (
                plt.show()   
            )
            return result
        
    except Exception as e:
        print("--- got error ---")
        print(e)

## 1.7. Plots a bivariate analysis
### Cần xem lại làm sao để outlier hiện thỉ
'''
Plots a bivariate analysis.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column
    :param y: str - column
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
'''
def bivariate_plot(dtf, x, y, max_cat=20, figsize=(10,5)):
    try:
        ## num vs num --> stacked + scatter with density
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
            breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
            groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, duplicates='drop')])[y].agg(['mean','median','size'])
            fig, ax = plt.subplots(figsize=figsize),
            #result = (
            fig.suptitle(x+"   vs   "+y, fontsize=20),
            groups[["mean", "median"]].plot(kind="line", ax=ax),
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True),
            ax.set(ylabel=y),
            ax.right_ax.set_ylabel("Observazions in each bin"),
            plt.show(),
            ### joint plot
            sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', height=int((figsize[0]+figsize[1])/2) ),
            plt.show()
            #)
            #return result
            
        ## cat vs cat --> hist count + hist %
        elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):  
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            order = dtf.groupby(x)[y].count().index.tolist(),
            a = dtf.groupby(x)[y].count().reset_index()
            a = a.rename(columns={y:"tot"})
            b = dtf.groupby([x,y])[y].count()
            b = b.rename(columns={y:0}).reset_index()
            b = b.merge(a, how="left")
            b["%"] = b[0] / b["tot"] *100
            #result = (
            fig.suptitle(x+"   vs   "+y, fontsize=20),
            ### count
            ax[0].title.set_text('count'),
            sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order, ax=ax[0]),
            ax[0].grid(True),
            ### percentage
            ax[1].title.set_text('percentage'),
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1]).get_legend().remove(),
            ax[1].grid(True),
            ### fix figure
            plt.close(2),
            plt.close(3),
            plt.show(),
            #)
            #return result
            
        ## num vs cat --> density + stacked + boxplot 
        else:
            if (utils_recognize_type(dtf, x, max_cat) == "cat"):
                cat,num = x,y
            else:
                cat,num = y,x
            fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(dtf[cat].unique()):
                sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ### stacked
            dtf_noNan = dtf[dtf[num].notnull()]  #can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf_noNan[num], q=np.linspace(0,1,11))
            tmp = dtf_noNan.groupby([cat, pd.cut(dtf_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[dtf_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(dtf[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
            ### boxplot   
            ax[2].title.set_text('outliers')
            sns.boxplot(x=cat, y=num, data=dtf, order=sorted(dtf[cat].unique()))
            ax[2].grid(True)
            ### fix figure
            #plt.close(2)
            #plt.close(3)
            #plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)

## 1.8. Plots a bivariate analysis using Nan and not-Nan as categories.
'''
Plots a bivariate analysis using Nan and not-Nan as categories.
'''
def nan_analysis(dtf, na_x, y, max_cat=20, figsize=(10,5)):
    dtf_NA = dtf[[na_x, y]]
    dtf_NA[na_x] = dtf[na_x].apply(lambda x: "Value" if not pd.isna(x) else "NA")
    result =  bivariate_plot(dtf_NA, x=na_x, y=y, max_cat=max_cat, figsize=figsize)
    return result

## 1.9. Plots a bivariate analysis with time variable.
'''
Plots a bivariate analysis with time variable.
'''
def ts_analysis(dtf, x, y, max_cat=20, figsize=(10,5)):
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        dtf_tmp = dtf.groupby(x)[y].sum()       
    else:
        dtf_tmp = dtf.groupby(x)[y].median()
    result = dtf_tmp.plot(title=y+" by "+x, figsize=figsize, grid=True)
    return result

# 1.10. plots multivariate analysis.
'''
plots multivariate analysis.
'''
def cross_distributions(dtf, x1, x2, y, max_cat=20, figsize=(10,5)):
    ## Y cat
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        
        ### cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            result = (
                sns.heatmap(cont_table, annot=True, fmt='.0f', cmap="YlGnBu", ax=ax, linewidths=.5).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
            )
            return result

        ### num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            result = (
                sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=figsize[1])
            )
            return result
        
        ### num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat,num = x1,x2
            else:
                cat,num = x2,x1
            fig, ax = plt.subplots(figsize=figsize)
            result = (
                sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1+'  vs  '+x2+'  (filter: '+y+')'),
                ax.grid(True)
            )
            return result
    ## Y num
    else:
        ### all num --> 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d'),
        plot3d = ax.scatter(xs=dtf[x1], ys=dtf[x2], zs=dtf[y], c=dtf[y], cmap='inferno', linewidth=0.5)
        result = (  
            fig.colorbar(plot3d, shrink=0.5, aspect=5, label=y),
            ax.set(xlabel=x1, ylabel=x2, zlabel=y),
            plt.show()
        )
        return result

##################################################
#          CORRELATION                           #
################################################## 

## 2.1.Computes the correlation matrix => Xem correclation dữa các biến xem thế nào và chỉ rõ nó ra => Những nhãn nào chưa có gán nhãn thì sẽ gán nhãn
'''
Computes the correlation matrix.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson" (numeric), "spearman" (categorical), "kendall"
    :param negative: bool - if False it takes the absolute values of correlation
    :param lst_filters: list - filter rows to show
    :param annotation: logic - plot setting
'''
def corr_matrix(dtf, method="pearson", negative=True, lst_filters=[], annotation=True, figsize=(10,5)):    
    ## factorize
    dtf_corr = dtf.copy()
    for col in dtf_corr.columns:
        if dtf_corr[col].dtype == "O":
            print("--- WARNING: Factorizing", dtf_corr[col].nunique(),"labels of", col, "---")
            dtf_corr[col] = dtf_corr[col].factorize(sort=True)[0]
    ## corr matrix
    dtf_corr = dtf_corr.corr(method=method) if len(lst_filters) == 0 else dtf_corr.corr(method=method).loc[lst_filters]
    dtf_corr = dtf_corr if negative is True else dtf_corr.abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_corr, annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    return dtf_corr


## 2.2. Computes the pps matrix.: Tính ma trận
'''
Computes the pps matrix.
'''
def pps_matrix(dtf, annotation=True, lst_filters=[], figsize=(10,5)):
    dtf_pps = ppscore.matrix(dtf) if len(lst_filters) == 0 else ppscore.matrix(dtf).loc[lst_filters]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_pps, vmin=0., vmax=1., annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title("predictive power score")
    return dtf_pps

    
'''
Computes correlation/dependancy and p-value (prob of happening something different than what observed in the sample)
'''
def test_corr(dtf, x, y, max_cat=20):
    ## num vs num --> pearson
    if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
        dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
        coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## cat vs cat --> cramer (chiquadro)
    elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
        cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
        chi2_test = scipy.stats.chi2_contingency(cont_table)
        chi2, p = chi2_test[0], chi2_test[1]
        n = cont_table.sum().sum()
        phi2 = chi2/n
        r,k = cont_table.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Cramer Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## num vs cat --> 1way anova (f: the means of the groups are different)
    else:
        if (utils_recognize_type(dtf, x, max_cat) == "cat"):
            cat,num = x,y
        else:
            cat,num = y,x
        model = smf.ols(num+' ~ '+cat, data=dtf).fit()
        table = sm.stats.anova_lm(model)
        p = table["PR(>F)"][0]
        coeff, p = None, round(p, 3)
        conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
        print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")
        
    return coeff, p


###############################################################################
#                       PREPROCESSING                                         #
###############################################################################

## 3.1. Chia dữ liệu train/ test
'''
Split the dataframe into train / test
shuffle: bool, default=True/ Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
return 2 tập dữ liệu dtf_train, dtf_test
'''
def dtf_partitioning(dtf, y, test_size=0.3, shuffle=True):
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    # Xem mức độ cân bằng dữ liệu
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    # Các giá trị trung bình của dữ liệu y với thuộc tính là regression
    print("y_train mean:", round(np.mean(dtf_train[y]),2), "| y_test mean:", round(np.mean(dtf_test[y]),2))
    # Các thuộc tính trong input
    print(dtf_train.shape[1], "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test


## 3.2. Xem mức độ cân bằng dữ liệu trong bài toán classification => Được xem bài toán mất cần bằng trong classification
### Cần phải hiểu hơn về mất cân bằng dữ liệu trong classification
### Vi sao chỉ có 2 thuật toán KNN và random mất cần bằng dữ liệu
'''
Rebalances a dataset with up-sampling and down-sampling.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - column to use as target 
    :param balance: str - "up", "down", if None just prints some stats
    :param method: str - "random" for sklearn or "knn" for imblearn
    :param size: num - 1 for same size of the other class, 0.5 for half of the other class
:return
    rebalanced dtf
'''
def rebalance(dtf, y, balance=None,  method="random", replace=True, size=1):
    ## check dùng để xem tỉ lệ các biến tỉ lệ % của mỗi biến
    print("--- situation ---")
    check = dtf[y].value_counts().to_frame()
    check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
    print(check)
    print("tot:", check[y].sum())

    ## sklearn đối với thuật toán Random thì nếu thiếu dữ liệu thì 
    if balance is not None and method == "random":
        ### set the major and minor class
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y]==major]
        dtf_minor = dtf[dtf[y]==minor]

        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   randomly replicate observations from the minority class (Overfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   randomly remove observations of the majority class (Underfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

    ## imblearn
    if balance is not None and method == "knn":
        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   create synthetic observations from the minority class (Distortion risk)")
            smote = imblearn.over_sampling.SMOTE(random_state=123)
            dtf_balanced, y_values = smote.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
       
        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   select observations that don't affect performance (Underfitting risk)")
            nn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=123)
            dtf_balanced, y_values = nn.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
        
    ## check rebalance
    if balance is not None:
        print("--- new situation ---")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced
    

## 3.3. Replace Na with a specific value or mean for numerical and mode for categorical. 
### Nếu NA trong dữ liệu thì nếu là num => Thay bằng giá trị trùng bình/ Nếu là categorical thì bằng giá trị mode
### Chỉ thay thế chó 1 giá trị duy nhất
'''
Replace Na with a specific value or mean for numerical and mode for categorical. 
'''
def fill_na(dtf, x, value=None):
    if value is None:
        value = dtf[x].mean() if utils_recognize_type(dtf, x) == "num" else dtf[x].mode().iloc[0]
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf, value
    else:
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf


## 3.4. Transforms a categorical column into dummy columns
### Dùng để chuyển về số bằng dummy trong Feature engineering
'''
Transforms a categorical column into dummy columns
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dropx: logic - whether the x column should be dropped
:return
    dtf with dummy columns added
'''
def add_dummies(dtf, x, dropx=False):
    dtf_dummy = pd.get_dummies(dtf[x], prefix=x, drop_first=True, dummy_na=False)
    dtf = pd.concat([dtf, dtf_dummy], axis=1)
    print( dtf.filter(like=x, axis=1).head() )
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf
    

# 3.5. Reduces the classes a categorical column: Giảm các cột thành 1 lớp phân loại
### 
'''
Reduces the classes a categorical column.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dic_clusters_mapping: dict - ex: {"min":[30,45,180], "max":[60,120], "mean":[]}  where the residual class must have an empty list
    :param dropx: logic - whether the x column should be dropped
'''
def add_feature_clusters(dtf, x, dic_clusters_mapping, dropx=False):
    dic_flat = {v:k for k,lst in dic_clusters_mapping.items() for v in lst}
    for k,v in dic_clusters_mapping.items():
        if len(v)==0:
            residual_class = k 
    dtf[x+"_cluster"] = dtf[x].apply(lambda x: dic_flat[x] if x in dic_flat.keys() else residual_class)
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf


## 3.6. Scales features MinMaxScaler
'''
Scales features.
'''
def scaling(dtf, y, scalerX=None, scalerY=None, fitted=False, task="classification"):
    scalerX = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerX is None else scalerX
    if fitted is False:
        scalerX.fit(dtf.drop(y, axis=1))
    X = scalerX.transform(dtf.drop(y, axis=1))
    dtf_scaled = pd.DataFrame(X, columns=dtf.drop(y, axis=1).columns, index=dtf.index)
    if task == "regression":
        scalerY = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerY is None else scalerY
        dtf_scaled[y] = scalerY.fit_transform(dtf[y].values.reshape(-1,1)) if fitted is False else dtf[y]
        return dtf_scaled, scalerX, scalerY
    else:
        dtf_scaled[y] = dtf[y]
        return dtf_scaled, scalerX


## 3.7. Computes all the required data preprocessing.
'''
Computes all the required data preprocessing.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable 
    :param processNas: str or None - "mean", "median", "most_frequent"
    :param processCategorical: str or None - "dummies"
    :param split: num or None - test_size (example 0.2)
    :param scale: str or None - "standard", "minmax"
    :param task: str - "classification" or "regression"
:return
    dictionary with dtf, X_names lsit, (X_train, X_test), (Y_train, Y_test), scaler
'''
def data_preprocessing(dtf, y, processNas=None, processCategorical=None, split=None, scale=None, task="classification"):
    try:
        dtf = pop_columns(dtf, [y], "front")
        
        ## missing
        ### check
        print("--- check missing ---")
        if dtf.isna().sum().sum() != 0:
            cols_with_missings = []
            for col in dtf.columns.to_list():
                if dtf[col].isna().sum() != 0:
                    print("WARNING:", col, "-->", dtf[col].isna().sum(), "Nas")
                    cols_with_missings.append(col)
            ### treat
            if processNas is not None:
                print("...treating Nas...")
                cols_with_missings_numeric = []
                for col in cols_with_missings:
                    if dtf[col].dtype == "O":
                        print(col, "categorical --> replacing Nas with label 'missing'")
                        dtf[col] = dtf[col].fillna('missing')
                    else:
                        cols_with_missings_numeric.append(col)
                if len(cols_with_missings_numeric) != 0:
                    print("replacing Nas in the numerical variables:", cols_with_missings_numeric)
                imputer = impute.SimpleImputer(strategy=processNas)
                imputer = imputer.fit(dtf[cols_with_missings_numeric])
                dtf[cols_with_missings_numeric] = imputer.transform(dtf[cols_with_missings_numeric])
        else:
            print("   OK: No missing")
                
        ## categorical data
        ### check
        print("--- check categorical data ---")
        cols_with_categorical = []
        for col in dtf.columns.to_list():
            if dtf[col].dtype == "O":
                print("WARNING:", col, "-->", dtf[col].nunique(), "categories")
                cols_with_categorical.append(col)
        ### treat
        if len(cols_with_categorical) != 0:
            if processCategorical is not None:
                print("...trating categorical...")
                for col in cols_with_categorical:
                    print(col)
                    dtf = pd.concat([dtf, pd.get_dummies(dtf[col], prefix=col)], axis=1).drop([col], axis=1)
        else:
            print("   OK: No categorical")
        
        ## 3.split train/test
        print("--- split train/test ---")
        X = dtf.drop(y, axis=1).values
        Y = dtf[y].values
        if split is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
            print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
            print("y_train mean:", round(np.mean(y_train),2), " | y_test mean:", round(np.mean(y_test),2))
            print(X_train.shape[1], "features:", dtf.drop(y, axis=1).columns.to_list())
        else:
            print("   OK: step skipped")
            X_train, y_train, X_test, y_test = X, Y, None, None
        
        ## 4.scaling
        print("--- scaling ---")
        if scale is not None:
            scalerX = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
            X_train = scalerX.fit_transform(X_train)
            scalerY = 0
            if X_test is not None:
                X_test = scalerX.transform(X_test)
            if task == "regression":
                scalerY = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
                y_train = scalerY.fit_transform(y_train.reshape(-1,1))
            print("   OK: scaled all features")
        else:
            print("   OK: step skipped")
            scalerX, scalerY = 0, 0
        
        return {"dtf":dtf, "X_names":dtf.drop(y, axis=1).columns.to_list(), 
                "X":(X_train, X_test), "y":(y_train, y_test), "scaler":(scalerX, scalerY)}
    
    except Exception as e:
        print("--- got error ---")
        print(e)

########################################################################################################################################################
#                                                   HANDLING IMBALANCE DATASET - MẤT CÂN BẰNG DỮ LIỆU                                                  #
########################################################################################################################################################


Handling_Imbalanced_Dataset = '''
    1. Đặt vấn đề
        - Bạn đang làm việc trên tập dữ liệu. Bạn tạo ra 1 mô hình phân loại (classification model) và nhận được độ chính xác 90% ngay lập tức. 
        Tốt quá, nhưng đến khi bạn hiểu sâu hơn một chút thì phát hiện ra rằng 90% dữ liệu thuộc về một lớp. Rồi xong ???!!!    
        => Đây là 1 ví dú về dữ liệu mất cân bằng và kết quả mà nó có thể gây ra
        - Giả sử bạn đang làm việc tại 1 công ty và bạn được yêu cầu tạo một mô hình, dựa trên các phép đo khác nhau theo ý của bạn, 
        mô hình sẽ dự đoán liệu sản phẩm có bị lỗi hay không. Bạn quyết định sử dụng trình phân loại yêu thích của mình huấn luyện cho nó trên dữ liệu và thật tuyệt: 
        bạn có độ chính xác 96.2 %
        - Sếp của bạn ngạc nhiên và quyết định sử dụng mô hình của bạn mà không cần kiểm tra thêm. Vài tuần sau anh ta 
        vào văn phòng của bạn và cho bạn một trận vì sự vô dụng của mô hình bạn đã làm. Sự thật là, mô hình của bạn tạo ra 
        không tim thấy bất kỳ sản phẩm bị lỗi nào kể từ khi nó được sử dụng trong sản xuất. Sau khi kiểm tra lại, 
        bạn phát hiện rằng chỉ có khoảng 3.8% sản phẩm do công ty của bạn sản xuất bị lỗi và mô hình của bạn luôn trả lời không bị lỗi, 
        dẫn đến độ chính xác 96.2%. Các kết quả mà bạn thu được là do bộ dữ liệu không cân bằng mà bạn đang làm việc. Ác mộng ????!!!
    2. Khi làm việc với dữ liệu cần chú ý
        - Phân loại là một trong những vấn đề machine learning phổ biến nhất. Cách tốt nhất để tiếp cận bất kỳ vấn đề phân loại là bắt đầu bằng việc phân tích và 
        khám phá bộ dữ liệu - Exploratory Data Analysis (EDA). Mục đích của công việc phân tích và khám phá dữ liệu là tạo ra càng nhiều insight và 
        thông tin về dữ liệu càng tốt. EDA cũng được sử dụng để tìm ra các vấn đề có thể tồn tại trong bộ dữ liệu
        - Một trong những vấn đề phổ biến được tìm thấy trong các bộ dữ liệu được sử dụng để phân loại là vấn đề về các lớp không cân bằng (imbalanced classes)
        - Nếu thuộc tính categorical có tính chất phân loại thì phải xem mức độ cân bằng dữ liệu không ? Nếu chêch lệch mất cân bằng thì phải xử lý nó.
    3. Mất cân bằng dữ liệu (Data imbalance) là gì ?    
        - Mất cân bằng dữ liệu thường phản ánh sự phân phối không đồng đều của các lớp trong một tập dữ liệu
        - Ví dụ: Trong 1 bộ dữ liệu phát hiện gian lận thẻ tín dụng, hầu hết các giao dịch thẻ tín dụng không phải là gian lận và rất ít là gian lận. 
        Đều này khiến chúng ta có tỷ lệ chênh lệch lớn (50:1, 1000:1...) giữa không gian lận: gian lận
    4. Cách để xem dữ liệu có mất cân bằng hay không ?
        - Như chúng ta có thể thấy, các giao dịch không gian lận vượt xa các giao dịch gian lận. 
        Nếu chung ta đào tạo 1 mô hình phân loại nhị phân mà không khăc phục vấn đề này, mô hình hoàn toàn sai lệch. 
        Nó cũng tác động đến mối tương quan giữa các tính năng trong dataset.
        - Trực quan hóa dữ liệu là một cách để xem có mất cân bằng dữ liệu không ?
        - Ví dụ: Bạn có thể có vấn đề phân loại 2 lớp (nhị phân ) với 1000 trường hợp (dòng).
        Tổng cộng có 800 trường hợp được gắn nhãn Class-1 và 200 trường hợp còn lại được gắn nhãn Class-2. 
        Đây là một bộ dữ liệu không cân bằng và tỷ lệ của các trường hợp Class-1 và Class-2 là 800:200 (hoặc 4:1)
    5. Mất cân bằng là một vấn đề phổ biến
        - Hầu hết các tập dữ liệu phân loại không có số lượng mẫu chính xác bằng nhau trong mỗi lớp, nhưng 1 sự chêch lệch nhỏ thường không quan trọng
        - Có những vấn đề mà sự mất cân bằng giữa các lớp rất phổ biến, hiễn nhiên
        - Khi có sự mất cân bằng lớp 4:1 như ví dụ trên => Có thể gây ra vấn đề
        - Có thể chấp nhận mất cân bằng những tỉ lệ nhỏ
    6. Nghịch lý chính xác (Accuracy Paradox)
        - Nghịch lý chính xác là tên gọi của trường hợp đo lường độ chính xác có kết quả chính xác tuyệt vời (ví dụ 90%), 
        nhưng độ chính xác này chỉ phản ánh cho một lớp, mà không phải là tất cả
        - Nó rất phổ biến, bởi vì độ chính xác phân loại thường là thước đo đầu tiên chung ta sử dụng khi đánh giá các mô hình phân loại
    7. Chiến thuật làm việc với dữ liệu mất cân bằng (Handling Imbalanced Dataset)
        Chiến lược 1: Thu thập thêm dữ liệu được khuyến khích sử dụng 
            - Đây là việc nên làm nhưng hầu như luôn bị bỏ qua
            - Hãy đặt cho mình câu hỏi: "Liệu có thể thu nhập thêm dữ liệu về vấn đề của mình không ?" => Dành thời gian suy nghĩ và trả lời câu hỏi này
            - Một tập dữ liệu lớn hơn có thể cho thấy một quan điểm khác biệt và có lẽ cần bằng hơn về các lớp
            - Chiến thuật cao nhất và mang lại hiệu quả cao nhất
            - Kinh phí và thời gian để thu thập dữ liệu rất tốn
        Chiến lược 2: Thay đổi performance Metric
            - Độ chính xác không phải là số liệu được dùng khi làm việc với bộ dữ liệu không cân bằng. Bới vì nó bị sai lệch
            - Có những số liệu đã được thiết kế để cho chúng ta biết cách chân thực hơn khi làm việc với các lớp không cân bằng
            - Có 2 phương pháp để thay đổi lượng chính xác của model
                a. Confusion Matrix:
                    * Phân tích các dự đoán bằng 1 bảng hiển thị các dự đoán chính xác (trên đường chéo) và các loại dự báo không chính xác 
                    ( các lớp dự đoán không chính xác đã được chỉ định)
                        + Precision: thước đo độ chính xác của phân loại
                        + Recall: Thước đo tính đầy đủ của phân loại
                        + F1 score (F-score): Trung bình của precision và recall
                        + Cách tốt và đơn giản thường được sử dụng khi xử lý vấn đề phân loại là confusion matrix. 
                        Số liệu này cung cấp một cái nhìn tổng quan thú vị về việc một mô hình đang hoạt động tốt hay không. 
                        Vì vậy nó là khời đầu tuyệt vời cho bất kỳ đánh giá mô hình phân loại
                        + Ý nghĩa công thức
                            1. Độ chính xác của mô hình về cơ bản là tổng số dự đoán đúng chia cho tổng số dư đoán
                            2. Độ chính xác (precision) của một lớp xác định mực độ tin cậy là kết quả khi mô hình trả lời một điểm thuộc về lớp đó
                            3. Recall của một lớp thể hiện mức độ tốt của mô hình có thể phát hiện lớp đó
                            4. F1-score của một lớp được tính bằng (2 x precision x recall)/(precision + recall),
                            nó là kết hợp precision và recall của một lớp trong một metric
                        + Đối với một lớp nhất định, các kết hợp recall và precision khác nhau có các ý nghĩa sau
                            1. High recall + High precision: lớp được xử lý hoàn hảo bởi mô hình
                            2. Low recall + High precision: mô hình có thể không phát hiện tốt lớp nhưng rất đáng tin cậy khi nó thực hiện
                            3. High recall + Low precision: Lớp được phát hiện tốt nhưng mô hình cũng bào gồm các điểm của các lớp khác trong đó
                            4. Low recall + Low precision: lớp được xử lý kém bới mô hình
                    b. ROC curves
                        + Giống như precision & recall, độ chính xác được chia thành độ nhạy (sensitivity) và 
                        độ đặc hiệu (specificity) và các mô hình có thể được chọn dựa trên ngưỡng cân bằng của các giá trị này
                        + Ý nghĩa của công thức
                            1. Hầu hết các phân loại tạo ra một điểm số, sau đó so với ngưỡng (theshold) để quyết định phân loại. 
                            Nếu 1 bộ phân loại tạo ra điểm số giữa 0.0 (chắc chắn nó là negative) và 
                            1.0 ( chắc chắn là positive với thông thường coi mọi thứ > 0.5 là positive)
                            2. Tuy nhiên bất kỳ ngưỡng nào cũng được áp dụng cho bộ dữ liệu (trong đó PP là positive population và NP là negative population) 
                            sẽ tạo ra true positive (TP), False positive (FP), true negative (TN) và false negative (FN). 
                            Chúng ta cần 1 phương pháp sẽ tính đến tất cả những con số này
            - Lấy mẫu lại bộ dữ liệu (Resampling Dataset)
                    a. Sau khi sử dụng Confusion matrix và ROC curver thấy không phù hợp thì chúng ta phải xem lại dữ liệu và 
                    có thể xem lại yêu cầu bài toán hoặc áp dụng 1 thuật toán khác
                    b. Chung ta có thể thay đổi tập dữ liệu mà ta sử dụng để xây dựng mô hình dự đoán để có dữ liệu cần bằng hơn
                    c. Thay đổi này gọi là lấy mẫu dữ liệu và có hai phương thức chính mà chúng ta có thể sử dụng
                        1. Thêm các bản sao của các thể hiện từ lớp đại diện dưới mức (under-represented class) được gọi là over-sampling 
                        (hoặc lấy mẫu hơn chính thức với sự thay thế)
                        2. Có thể xóa các thể hiện khỏi lớp được đại diện quá mức (over-represented class) được gọi là oversampling
                    d. Lấy dữ liệu giả lâp từ thư viên để làm giả lập cho cân bằng dữ liệu
                    e. Những cách tiếp cận này thường rất dễ thực thi và nhanh chóng thực thi => Đây cũng là giải pháp khởi đầu tốt
                    f. Trên thực tế, chung ta nên sử dụng cả 2 cách tiếp cận trên cho tất cả các bộ dữ liệu mất cân bằng, 
                    để xem liệu nó có giúp chung ta tăng cường các đo lường chính xác không
            - Một số nguyên tắc
                    a. Cân nhắc kiểm tra việc lấy mẫu under-sampling khi có nhiều dữ liệu (hàng chục hoặc hàng trăm nghìn trường hợp trở lên)
                    b. Cân nhắc kiểm tra việc lấy mẫu over-sampling khi không có nhiều dữ liệu (hàng nghìn trường hợp hoặc ít hơn)
                    c. Xem xét thử nghiệm các scheme lấy mẫu ngẫu nhiên (random) và không ngẫu nhiên ( non-random) (VD: Phân từng)
                    d. Xem xét thử nghiệm các tỷ lệ mẫu khác nhau (VD: ngoài tỷ lệ 1:1 trong bài toán phân loại nhị phân, hãy thử các tỉ lệ khác)
            - Thử các thuật toán ML khác nhau
                    a. Không nên sử dụng thuật toán yêu thích của mình cho mọi vấn đề. Ít nhất chung ta nên kiểm tra cùng lúc 
                    (Spot checking) nhiều loại thuật toán khác nhau cho một vấn đề nhất định
                    b. Gợi ý: Decision - cây quyết định thường hoạt động tốt trên các bộ dữ liệu mất cân bằng. 
                    Các quy tắc phân tách dựa vào biến lớp được sử dụng trong việc tạo cây, có thể ràng buộc các lớp được sử lý. 
                    Cũng có thể sử dụng Random Forest trong trường hợp này

'''
