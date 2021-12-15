##########################################################################################################################################################
# THƯ VIỆN
##########################################################################################################################################################
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

import pickle
#########################################################################################################################################################
# COLLABORATIVE FILTERING
#########################################################################################################################################################
import lib.step15_collaborative_filtering as clf

def pyspark_collaborative_filtering(fea_customer_id_number):
    #Bước 1: Create the SparkSession Object
    spark = clf.get_spark_session(title='Recommender Systems - Hệ thống đề xuất')
    #Bước 2: Read the Dataset: Chung ta khi load và đọc dữ liệu với Spark phải sử dụng dataframe.
    file_path = 'data_analysis/data_merge.csv'
    df = clf.load_data(file_path=file_path)
    lst = ['fea_item_id','fea_name','fea_customer_id','fea_rating_y','image','url','brand','list_price','price','rating']
    data_sub = clf.select_data(df=df, lst=lst)
    #Bước 3: Exploratory Data Analysis: Khám phá dữ liệu bới vì xem xét dữ liệu, điều kiện dữ liệu và đếm xem có bao nhiều mã hàng, bao nhiêu sao đánh giá.
    #Bước 4: Feature Engineering: Chúng tôi sẽ convert những column categorical sang numerical values using StringIndexex
    indexed = clf.feature_engineering_stringIndexer(data_sub=data_sub, inputCol="fea_name", outputCol="title_new")
    #Bước 5: Splitting the Dataset: Chung ta có thể so sánh dữ liệu xây dựng mô hình đề xuất, chung có thể chia dữ liệu theo dữ liệu training và test và chung ta có thể chia trong khoảng 75 đến 25 chỉ số train mode và test accuracy
    train, test =  clf.random_split(data_sub=data_sub, randoms = [0.75,0.25])
    train.count(), test.count()
    #Bước 6: Build and Train Recommender Model: Chúng ta có thể import ALS thư viện Pyspark machine learning và xây dựng mô hình training dữ liệu và có rất nhiều parameter nhưng có tuned cải thiện hiệu suất mô hình trong đo: có 2 para quan trọng đó là:  nonnegative =‘True’: Nó không thêm negative ratings in recommendations và coldStartStrategy=‘drop’ to prevent any NaN ratings predictions
    rec_model = clf.als_model(train=train, maxIter=15, userCol='fea_customer_id',itemCol='fea_item_id',ratingCol='fea_rating_y', nonnegative=True, coldStartStrategy="drop")
    #Bước 7: Predictions and Evaluation on Test Data: Hoàn thành 1 phần bài tập là kiểm tra hiệu quả của mô hình hoặc test dữ liệu và chúng ta sử dụng các hàm làm dự đoán trên tập test data và RegressionEvaluate để check the RMSE value của tập dữ liệu
    predicted_ratings = clf.transform_test(rec_model = rec_model, test = test)
    rmse = clf.evaluator_als(predicted_ratings)
    #Bước 8: Recommend Top item That Active User Might Like: Sau khi kiểm tra hiểu quả về mô hình và tuning the hyperparameters và chúng ta có thể duy chuyển để xuất theo top items tới user id đó và họ có thể nhìn và thích
    fea_customer_id_number = fea_customer_id_number
    recommendations = clf.recomend_top_item(data_sub, rec_model, fea_customer_id_number, fea_item_id = 'fea_item_id',\
        fea_customer_id='fea_customer_id',how='left', \
        lst_ = ['fea_item_id','fea_name','fea_customer_id','fea_rating_y','image','url','brand','list_price','price','rating'])
    recommendations_pandas = recommendations.toPandas()
    
    dir_file2 = "data_analysis/ProductRaw_processing_2.csv"
    df2 = pd.read_csv(dir_file2)
    df2 = df2.rename(columns={'fea_product_id': 'fea_item_id'})
    data = pd.merge(recommendations_pandas, df2, on ='fea_item_id', how ='inner')
    lst2 = ['fea_customer_id', 'prediction', 'fea_item_id','name','description', 'rating', 'price', 'list_price', 'brand','group', 'url', 'image' ]
    data_1 = data[lst2]
    data_2 = data_1.head(10)
    return data_2, rmse
    
dir_file3 = "data_analysis/fea_customer_id.csv"
df3 = pd.read_csv(dir_file3)



###########################################################################################################################################################
# CONTENT BASED FILTERING
###########################################################################################################################################################

data_content=pickle.load(open('item_list.pkl','rb'))
vector=pickle.load(open('similarity.pkl','rb'))

def recommend(content):
    content=data_content[data_content.fea_name==content].index.values[0]
    
    df_recommnend=pd.DataFrame(enumerate(vector[content])).drop(0,axis='columns').sort_values(by=1,ascending=False)
    df_recommnend['Names']=list(map(lambda x: str(np.squeeze(data_content[data_content.index==x]['fea_name'].values)),df_recommnend.index.values))
    df_recommnend['id']=list(map(lambda x: int(np.squeeze(data_content[data_content.index==x]['fea_item_id'].values)),df_recommnend.index.values))

    df_recommnend['url']=list(map(
        lambda x: np.squeeze(data_content[data_content.index==x]['url'].values),df_recommnend.index.values))
    df_recommnend['image']=list(map(
        lambda x: np.squeeze(data_content[data_content.index==x]['image'].values),df_recommnend.index.values))
    df_recommnend['rating']=list(map(
        lambda x: int(np.squeeze(data_content[data_content.index==x]['rating'].values)),df_recommnend.index.values))
    df_recommnend['price']=list(map(
        lambda x: int(np.squeeze(data_content[data_content.index==x]['price'].values)),df_recommnend.index.values))
    df_recommnend['brand']=list(map(
        lambda x: np.squeeze(data_content[data_content.index==x]['brand'].values),df_recommnend.index.values))
    df_recommnend = df_recommnend.reset_index()
    df_recommnend = df_recommnend.drop(['index'], axis=1)
    
    return df_recommnend

###########################################################################################################################################################
# STREAMLIT
###########################################################################################################################################################
menu = ['Tiki','Tổng quan','Giới thiệu mô hình - Content based fitering','Giới thiệu mô hình - Collaborative filtering', 'Thực hiện bài toán', 'Đế xuất dựa trên nội dung', 'Đề xuất dựa trên sản phẩm', 'Kết luật và hướng phát triển hệ thống đề xuất']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Tiki':
    st.markdown("<h1 style='text-align: center; color: Coral;'>TIKI NIỀM TỰ HÀO VIỆT NAM - CHÀO MỪNG ĐẾN HỆ THỐNG TIKI</h1>", unsafe_allow_html=True)
    st.image('picture/tiki.PNG')
    st.markdown("- Với phương châm hoạt động “Tất cả vì Khách Hàng”, Tiki luôn không ngừng nỗ lực nâng cao chất lượng dịch vụ và sản phẩm, từ đó mang đến trải nghiệm mua sắm trọn vẹn cho Khách Hàng Việt Nam với dịch vụ giao hàng nhanh trong 2 tiếng và ngày hôm sau TikiNOW lần đầu tiên tại Đông Nam Á, cùng cam kết cung cấp hàng chính hãng với chính sách hoàn tiền 111% nếu phát hiện hàng giả, hàng nhái.")
    st.markdown("- Thành lập từ tháng 3/2010, Tiki.vn hiện đang là trang thương mại điện tử lọt top 2 tại Việt Nam và top 6 tại khu vực Đông Nam Á.")
    st.markdown("- Tiki lọt Top 1 nơi làm việc tốt nhất Việt Nam trong ngành Internet/E-commerce 2018 (Anphabe bình chọn), Top 50 nơi làm việc tốt nhất châu Á 2019 (HR Asia bình chọn).")
    st.markdown("[Website](https://tiki.vn/)")
    st.write(" ")

elif choice == 'Tổng quan':
    st.image('picture/teamwork.jpg')
    st.markdown("<h1 style='text-align: center; color: Coral;'>HỆ THỐNG ĐỀ XUẤT SẢN PHẨM CHO NGƯỜI DÙNG</h1>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<h2 style='text-align: left; color: Yellow;'>TỔNG QUAN HỆ THỐNG ĐỀ XUẤT NGƯỜI DÙNG</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Giới thiệu</h3>", unsafe_allow_html=True)
    st.markdown("- Tiki là một hệ sinh thái thương mại 'all in one', trong đó có tiki.vn, là một website thương mại điện tự đứng top 2 của Vietnam và top 6 khu vực Đông Nam Á")   
    st.markdown("- Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.")
    st.markdown("- Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Vì sao có dự án nào ?</h3>", unsafe_allow_html=True)
    st.markdown("- Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng. => Xây dựng các mô hình đề xuất")   
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>3. Dữ liệu cung cấp</h3>", unsafe_allow_html=True)
    st.markdown("- Dữ liệu được cung cấp sẵn gồm có các tập tin: ProductRaw.csv, ReviewRaw.csv chứa thông tin sản phẩm, review và rating cho các sản phẩm thuộc các nhóm hàng hóa như Mobile_Tablet, TV_Audio, Laptop, Camera, Accessory")   
    st.write(" ")
    
    st.markdown("<h3 style='text-align: left; color: Aqua;'>4. Đặt ra yêu cầu với bài toán</h3>", unsafe_allow_html=True)
    st.image('picture/compare.png')
    st.markdown("- Bài toán 1: Đề xuất người dùng với content - based filtering")
    st.markdown("- Bài toán 2: Đề xuất người dùng với Collaborative filtering")
    st.write(" ")

elif choice == 'Giới thiệu mô hình - Content based fitering':

    st.markdown("<h1 style='text-align: center; color: Coral;'>CONTENT BASED FILTERING</h1>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Những gì content based filtering làm được ?</h3>", unsafe_allow_html=True)
    st.image('picture/hay.png')
    st.markdown("- Đề xuất các mục tương tự cho người dùng mà người dùng đã thích trong quá khứ - This type of RS recommends similar items to the users that the user has liked in the past")
    st.markdown("- Vì vậy, toàn bộ ý tưởng là tính toán điểm tương đồng giữa hai mục bất kỳ và được đề xuất cho người dùng dựa trên hồ sơ về sở thích của người dùng - So, the whole idea is to calculate a similarity score between any two items and recommended to the user based upon the profile of the user’s interests")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Hoạt động của content based filtering như thế nào ?</h3>", unsafe_allow_html=True)
    st.markdown("- Bước 1: Lọc dựa trên nội dung")
    st.markdown("- Bước 2: Chúng đề xuất các mục tương tự dựa trên một nội dung cụ thể")
    st.markdown("- Bước 3: Hệ thống này sử dụng siêu dữ liệu nội dung, chẳng hạn như thể 1 sản phẩm cụ thể, loại sản phậm, những nội dung cần tìm kiếm v.v. để đưa ra các đề xuất cho người dùng")
    st.markdown("- Ý tưởng chung đằng sau các hệ thống giới thiệu này là nếu một người thích một mặt hàng cụ thể, họ cũng sẽ thích một mặt hàng tương tự với nó")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>3. Similarity Score</h3>", unsafe_allow_html=True)
    st.image('picture/Similarity_Score.png')
    st.markdown("- Làm thế nào để nó quyết định mặt hàng nào giống với mặt hàng mà người dùng thích nhất ?")
    st.markdown("- Thứ nhất: Ở đây chúng tôi sử dụng điểm tương đồng. Nó là một giá trị số nằm trong khoảng từ 0 đến 1, giúp xác định xem hai mục tương tự nhau ở mức độ nào trên thang điểm từ 0 đến 1.")
    st.markdown("- Thứ hai: Điểm tương tự này thu được khi đo mức độ giống nhau giữa các chi tiết văn bản của cả hai mục.")
    st.markdown("- Vì vậy, điểm tương tự là thước đo mức độ giống nhau giữa các chi tiết văn bản nhất định của hai mục. Điều này có thể được thực hiện bằng tính tương tự cosine.")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>4. Cosine Similarity hoạt động như thế nào ?</h3>", unsafe_allow_html=True)
    st.image('picture/Similarity_Score_2.png')
    st.markdown("- Cosine Similarity là một số liệu được sử dụng để đo lường mức độ tương tự của các tài liệu bất kể kích thước của chúng")
    st.markdown("- Về mặt toán học, nó đo cosin của góc giữa hai vectơ được chiếu trong không gian đa chiều. Sự giống nhau về cosin là có lợi vì ngay cả khi hai item tương tự cách xa nhau bằng khoảng cách Euclide (do kích thước của tài liệu), rất có thể chúng vẫn được định hướng gần nhau hơn. Góc càng nhỏ thì độ tương đồng cosin càng cao")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>5. Hiện thỉ bảng dữ liệu của bài toán ?</h3>", unsafe_allow_html=True)
    selected_movie = st.selectbox('Lựa chọn 1 item bất kỳ',(list(data_content.fea_name.values)))
    df_recommnend = recommend(selected_movie)
    test = df_recommnend.astype(str)
    st.dataframe(test)
    

elif choice == 'Giới thiệu mô hình - Collaborative filtering':

    st.markdown("<h1 style='text-align: center; color: Coral;'>COLLABORATIVE FILTERING</h1>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Collaborative filtering là gì ?</h3>", unsafe_allow_html=True)
    st.image('picture/Collaborative-filtering.jpg')
    st.markdown("- Là một tập hợp con các thuật toán sử dụng rộng rãi người dùng và mục khác cùng với xếp hạng và lịch sử người dùng mục tiêu của họ để đề xuất một mục mà người dùng mục tiêu không có xếp hạng")
    st.markdown("- Giả định cơ bản đằng sau cách tiếp cận này là những người dùng khác ưa thích các mặt hàng có thể được sử dụng để giới thiệu một mặt hàng cho người dùng chưa từng xem hoặc đã mua hàng trước đó.")
    st.markdown("- CF khác với các phương pháp dựa trên nội dung ở chỗ người dùng hoặc bản thân mặt hàng không đóng vai trò trong đề xuất mà là cách (xếp hạng) và người dùng (người dùng) đánh giá một mặt hàng cụ thể")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Thuật toán Collaborative filtering</h3>", unsafe_allow_html=True)
    st.image('picture/thuat_toan.png')
    st.markdown("- Các thuật toán dựa trên mô hình xây dựng một mô hình từ hành vi trước đây của người dùng, sau đó sử dụng mô hình đó để đề xuất các mặt hàng cho bất kỳ người dùng nào")
    st.markdown("- Chúng có hai ưu điểm chính so với các phương pháp tiếp cận dựa trên bộ nhớ: chúng có thể cung cấp các đề xuất cho người dùng mới và chúng có thể cung cấp các đề xuất tức thì, vì hầu hết việc tính toán được chuyển sang giai đoạn xử lý trước của quá trình tạo mô hình. Các thuật toán dựa trên bộ nhớ kết hợp việc tạo mô hình và các đề xuất tức thì thành một thủ tục duy nhất")
    st.markdown("- Các thuật toán yếu tố tiềm ẩn giải thích sở thích của người dùng bằng cách mô tả đặc điểm của sản phẩm và người dùng với các yếu tố được tự động suy ra từ phản hồi của người dùng.")
    st.markdown("- Ở dạng cơ bản, phân tích nhân tử ma trận đặc trưng cho các mặt hàng và người dùng bằng vectơ của các yếu tố được suy ra từ các mẫu sử dụng mặt hàng. Sự tương ứng cao giữa các yếu tố mặt hàng và người dùng dẫn đến một đề xuất. Các phương pháp này đã trở nên phổ biến vì chúng kết hợp độ chính xác của dự đoán với khả năng mở rộng tốt.")
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>3. Thuật toán ALS - Alternating Least Squares</h3>", unsafe_allow_html=True)
    st.image('picture/ALS.png')
    st.markdown("- Thuật toán ALS sẽ phát hiện ra các yếu tố tiềm ẩn giải thích người dùng quan sát được xếp hạng mặt hàng và cố gắng tìm trọng số yếu tố tối ưu để giảm thiểu bình phương nhỏ nhất giữa xếp hạng dự đoán và thực tế.")
    st.write(" ")

elif choice == 'Thực hiện bài toán':

    st.markdown("<h1 style='text-align: center; color: Coral;'>CÁC BƯỚC THỰC HIỆN</h1>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. Thực hiện content based fitering và Project design</h3>", unsafe_allow_html=True)
    st.image('picture/content_based_fitering.png')
    st.write(" ")

    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Thực hiện Collaborative filtering và Project design</h3>", unsafe_allow_html=True)
    st.image('picture/Collaborative_filtering.png')
    st.write(" ")

elif choice == 'Đế xuất dựa trên nội dung':
    st.image('picture/tiki2.PNG')
    st.markdown("<h3 style='text-align: center; color: Coral;'>RECOMMENDATION SYSTEM BY BIG TEAM</h3>", unsafe_allow_html=True)
    st.write(" ")
    selected_movie = st.selectbox('Chọn sản phẩm đi nào bạn',(list(data_content.fea_name.values)))
    df_recommnend = recommend(selected_movie)
    if st.button("Hãy nhấn vào tôi đi nào 🤡"):
        st.markdown("<h3 style='text-align: left; color: Aqua;'>BẢNG SẢN PHẨM</h3>", unsafe_allow_html=True)
        df_recommnend.reset_index()
        product_img = df_recommnend['image'].tolist()
        #for i in range(1,len(df_recommnend.Names.values[:5])):
        # get movie id and movie title
        item_id = df_recommnend['id'][0]
        item_name = df_recommnend['Names'][0]
        product_img_1 = product_img[0]
        col11,col12 = st.columns(2)
        with col11:
            st.markdown('![Foo]('+product_img[0]+')')
        with col12:
            st.subheader("TITLE: " + selected_movie.upper())
            st.write("BRAND: " + df_recommnend['brand'][0]) 
            st.write("VOTES: " + str(df_recommnend['rating'][0]))
            st.write("PRICE: " + "{:0,}".format(float(df_recommnend['price'][0]))+" VNĐ")
        
        st.header("Recommendations")
        col1,col2 = st.columns(2)
        col4,col5 = st.columns(2)
        col6,col7 = st.columns(2)
        
        columns = [col1,col2,col4,col5,col6,col7]
        for i in range(len(columns)):
            with columns[i]:
                st.markdown('![Foo]('+product_img[i]+')')
                st.write(df_recommnend['Names'][i].upper())
                st.write("VOTES: " + str(df_recommnend['rating'][i]))
                st.write("BRAND: " + df_recommnend['brand'][i]) 
                st.write("PRICE: " + "{:0,}".format(float(df_recommnend['price'][i]))+" VNĐ")
                st.markdown("[Website]("+df_recommnend['url'][i]+")")
    
        st.markdown("<h3 style='text-align: left; color: Aqua;'>HIỆN THỊ BẢNG DỮ LIỆU</h3>", unsafe_allow_html=True)
        test = df_recommnend.astype(str)
        st.dataframe(test)
    
elif choice == 'Đề xuất dựa trên sản phẩm':
    st.image('picture/tiki2.PNG')
    #st.markdown("<h3 style='text-align: center; color: Coral;'>RECOMMENDATION SYSTEM BY BIG TEAM</h3>", unsafe_allow_html=True)
    #st.title("Movie Recommendation Engine")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Item Recommendation Engine App by Big team </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    test_user =st.text_input("Enter the User for whom you wanna see top 10 recommendations：",16762580)
    result=""
    
    if st.button("Hãy nhấn vào tôi đi nào 🤡"):
        data, rmse=pyspark_collaborative_filtering(test_user)
        product_img = data['image'].tolist()
        st.text('Top 10 items recommendations for user id'+' '+str(test_user)+' '+'are:')
        st.markdown("<h3 style='text-align: left; color: Aqua;'>HIỆN THỊ BẢNG DỮ LIỆU</h3>", unsafe_allow_html=True)
        st.write("RMSE = {}".format(round(rmse, 2)))
        test = data.astype(str)
        st.dataframe(test)

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown("<h3 style='text-align: left; color: Aqua;'>SẢN PHẨM ĐỀ XUẤT NGƯỜI DÙNG</h3>", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)
        col5,col6 = st.columns(2)
        col7,col8 = st.columns(2)
        col9,col10 = st.columns(2)
        
        columns = [col1,col2,col3,col4,col5,col6,col7,col8,col9,col10]
        
        for i in range(len(columns)):
            with columns[i]:
                #lst2 = ['fea_customer_id', 'prediction', 'fea_item_id','name','description', 'rating', 'price', 'list_price', 'brand','group', 'url', 'image' ]
                st.markdown('![Foo]('+product_img[i]+')')
                st.write(data['name'][i].upper())
                st.write("VOTES: " + str(data['rating'][i]))
                st.write("BRAND: " + data['brand'][i]) 
                st.write("PRICE: " + "{:0,}".format(float(data['price'][i]))+" VNĐ")
                st.markdown("[Website]("+data['url'][i]+")")

       
        

elif choice == 'Kết luật và hướng phát triển hệ thống đề xuất':
    #st.image('picture/Ket_luan.jpg')
    st.markdown("<h1 style='text-align: center; color: Coral;'>KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: Aqua;'>1. SWOT</h3>", unsafe_allow_html=True)
    st.write(" ")
    st.image('picture/SWOT.png')
    
    st.markdown("<h3 style='text-align: left; color: Aqua;'>2. Tài liệu tham khảo</h3>", unsafe_allow_html=True)
    st.markdown("http://www.salemmarafi.com/code/collaborative-filtering-with-python/")
    st.markdown("https://www.mapr.com/blog/inside-look-at-components-of-recommendation-engine")
    st.markdown("Book: Pyspark of Wenqiang Feng")
    st.markdown("Book: Machine Learning with PySpark With Natural Language Processing and Recommender Systems by Pramod Singh (z-lib.org)")
    st.markdown("Book: User Factors in Recommender Systems_ Case Studies in e-Commerce, News Recommending, and ... ( PDFDrive )")
    