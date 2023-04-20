import pandas as pd
import numpy as np
import seaborn as sb
from IPython.core import display as ICD
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

                            ##############################################################
class DataFrame_Analyze:
    
    def __init__(self,dataframe):
        self.dataframe = dataframe
#0/--------------------------------------------------------------------------    
    def show(self):
        ICD.display(self.dataframe)
#1/--------------------------------------------------------------------------     
    def show_head(self):
        ICD.display(self.dataframe.head())
#2/--------------------------------------------------------------------------     
    def show_tail(self):
        ICD.display(self.dataframe.tail())
#3/--------------------------------------------------------------------------         
    def filter_column(self,types = object):
        return self.dataframe.select_dtypes(types)
#4/--------------------------------------------------------------------------     
    def get_central_tendency(self):
        def cal_mode(A):
            mode = []
            for i in A.value_counts()[A.value_counts() == max(A.value_counts())].index :
                mode.append(i)
            return mode
        # đặt biến
        all_mean = []
        all_median = []
        all_mode = []
        all_min = []
        all_max = []
        all_field = []
        # chạy vòng lặp
        if isinstance(self.dataframe, pd.DataFrame):
            for i in self.dataframe.columns :
                all_mean.append(round(self.dataframe[i].mean(),2))
                all_median.append(round(self.dataframe[i].median(),2))
                all_mode.append(cal_mode(self.dataframe[i]))
                all_min.append(min(self.dataframe[i]))
                all_max.append(max(self.dataframe[i]))
                all_field.append(i)
            
            return pd.DataFrame({'variable':all_field,'mean':all_mean,
                             'median':all_median,'mode':all_mode,'min':all_min,'max':all_max})
        else :
            all_mean.append(round(self.dataframe.mean(),2))
            all_mode.append(cal_mode(self.dataframe))
            all_min.append(min(self.dataframe))
            all_max.append(max(self.dataframe))
            all_field.append(self.dataframe.name)
            return pd.DataFrame({'variable':all_field,'mean':all_mean,
                             'median':all_median,'mode':all_mode,'min':all_min,'max':all_max})
            
#5----------------------------------------------------------------------------
    def get_dispersion(self):
        all_range = []
        all_q1 = []
        all_q3 = []
        all_iqr = []
        all_var = []   
        all_std = []
        all_skew = []
        all_kurtosis = []
        all_field = []
        if isinstance(self.dataframe, pd.DataFrame):
            for i in self.dataframe.columns:
                all_range.append(round(max(self.dataframe[i]) - min(self.dataframe[i])))
                all_q1.append(round(self.dataframe[i].quantile(0.25),2))
                all_q3.append(round(self.dataframe[i].quantile(0.75),2))
                all_iqr.append(round(self.dataframe[i].quantile(0.75),2)-round(self.dataframe[i].quantile(0.25),2))
                all_var.append(round(self.dataframe[i].std(),2))
                all_std.append(round(self.dataframe[i].var(),2))
                all_skew.append(round(self.dataframe[i].skew(),2))
                all_kurtosis.append(round(self.dataframe[i].kurtosis()+3,2))
                all_field.append(i)
            return pd.DataFrame({'variable':all_field,'range':all_range,'q1':all_q1,'q3':all_q3,
                'iqr':all_iqr, 'var': all_var,'std': all_std,'skew':all_skew, 'kurtosis':all_kurtosis})
        else :
            all_range.append(round(max(self.dataframe) - min(self.dataframe)))
            all_q1.append(round(self.dataframe.quantile(0.25),2))
            all_q3.append(round(self.dataframe.quantile(0.75),2))
            all_iqr.append(round(self.dataframe.quantile(0.75),2)-round(self.dataframe.quantile(0.25),2))
            all_var.append(round(self.dataframe.std(),2))
            all_std.append(round(self.dataframe.var(),2))
            all_skew.append(round(self.dataframe.skew(),2))
            all_kurtosis.append(round(self.dataframe.kurtosis()+3,2))
            all_field.append(self.dataframe.name)
            return pd.DataFrame({'variable':all_field,'range':all_range,'q1':all_q1,'q3':all_q3,
                'iqr':all_iqr, 'var': all_var,'std': all_std,'skew':all_skew, 'kurtosis':all_kurtosis})
            
#6------------------------------------------------------------------------------
    def visualize_hist_box_plot(self,remove_outlier = False) :
        if remove_outlier == False :
            data = self.dataframe
            for i in self.dataframe.columns:
                print('BIẾN {}'.format(i))
                DataFrame_Analyze(DataFrame_Analyze(data[[i]]).get_central_tendency()).show()
                DataFrame_Analyze(DataFrame_Analyze(data[[i]]).get_dispersion()).show()
                plt.figure(figsize=(6,4))
                plt.subplot(1,2,1)
                data[i].plot.hist()
                plt.title(f'Hist plot of {data[i].name}  :')
                plt.subplot(1,2,2)
                data[i].plot.box()
                plt.title(f'Box plot of {data[i].name}')
                plt.show()
                print('******************************************************************************************')
        else :
            for i in self.dataframe.columns:
                data = DataFrame_Analyze(self.dataframe[[i]].dropna()).remove_outlier()
                print('BIẾN {}'.format(i))
                DataFrame_Analyze(DataFrame_Analyze(data).get_central_tendency()).show()
                DataFrame_Analyze(DataFrame_Analyze(data).get_dispersion()).show()
                plt.figure(figsize=(6,4))
                plt.subplot(1,2,1)
                data[i].plot.hist()
                plt.title(f'Hist plot of {data[i].name}  :')
                plt.subplot(1,2,2)
                data[i].plot.box()
                plt.title(f'Box plot of {data[i].name}')
                plt.show()
                print('******************************************************************************************')
            
            
#7---------------------------------------------------------------------------------    
    def remove_outlier(self):
        q1 = np.quantile(self.dataframe,0.25)
        q3 = np.quantile(self.dataframe,0.75)
        iqr = q3-q1
        h_limit = q3 + iqr * 1.5  
        l_limit = q1 - iqr * 1.5
        return self.dataframe[(self.dataframe <= h_limit)|(self.dataframe >= l_limit)]
    
#8----------------------------------------------------------------------------------
    def visualize_bar_plot(self):
        for i in self.dataframe.columns:
            print('BIEN [{}]'.format(i))
            self.dataframe[i].value_counts().plot.bar()
            ICD.display(pd.DataFrame(self.dataframe[i].value_counts()))
            plt.show()
            print('*****************************************************************************************')

            
#9----------------------------------------------------------------------------------
    def tw_variable(self):
        n = len(self.dataframe.columns)
        tw_variable = []
        for i in range(n):
            for j in range(i+1,n):
                tw_variable.append([self.dataframe.columns[i],self.dataframe.columns[j]])
        return tw_variable
#10----------------------------------------------------------------------------------
    def analyze_category_vs_category(self,alpha = 0.05):
        for i in DataFrame_Analyze(self.dataframe).tw_variable():
            table = pd.crosstab(self.dataframe[i[0]],self.dataframe[i[1]])
            if len(table) != 0 :    
                print(f'Biến [{i[0]}] và biến [{i[1]}]:')
                ICD.display(table)
                if DataFrame_Analyze(table).chi_2_test(alpha)[0] == True:
                    print('Hai biến độc lập')
                else :
                    print('Hai biến phụ thuộc nhau')
                table.plot.bar(stacked=True)
                plt.show()
            print('*****************************************************************************************')
#11-----------------------------------------------------------------------------------
    def chi_2_test(self,alpha = 0.05):
        X, p, dof, expctd = chi2_contingency(self.dataframe)
        c_X = chi2.ppf(1-alpha,dof)
        if X>c_X:
            return(False,X,c_X,p,dof,expctd) # Bác bỏ H0 -- H0:Phụ thuộc
        else:
            return(True,X,c_X,p,dof,expctd) # Chấp nhận H0
#12-------------------------------------------------------------------------------------
    def anova_test(self,function,category,continous):
        if len(category) > 1 :
            model = ols(function, data=self.dataframe[[category[0],category[1],continous]]).fit()
            anova_table = sm.stats.anova_lm(model, type=2)
            ICD.display(anova_table)
        else : 
            model = ols(function, data=self.dataframe[[category[0],continous]]).fit()
            anova_table = sm.stats.anova_lm(model, type=2)
            ICD.display(anova_table)                
#13--------------------------------------------------------------------------------------
    def analyze_continous_vs_categories(self,category_vars_name,continous_vars_name,alpha = 0.05,groupby = 0):
        #Tách chuỗi :
        def match(data):
            n = len(data)
            _list = []
            for i in range(n):
                for j in range(i+1,n):
                    _list.append([data[i],data[j]])
            return _list
        #Chạy anova :            
        for j in continous_vars_name :
            print(f'                              BIẾN LIÊN TỤC {[j]}:')
            print('')
            if len(category_vars_name) == 1:
                print(f'Biến category {category_vars_name}, biến liên tục {[j]}')
                function = f'{j} ~ C({category_vars_name[0]})'
                sb.boxplot(x=category_vars_name[0], y=j, data=self.dataframe)
                plt.show()
                DataFrame_Analyze(self.dataframe).anova_test(function,category_vars_name,j)
                print('****************************************************************************')
                   
            elif len(category_vars_name) > 1  :
                for i in match(category_vars_name) :
                    print(f'Biến category {[i[groupby]]},{[i[1-groupby]]}, biến liên tục {[j]}')
                    function = f"{j} ~ C({i[groupby]}) + C({i[1-groupby]}) + C({i[groupby]}):C({i[1-groupby]})"
                    sb.boxplot(x=i[groupby], y=j,hue = i[1-groupby], data=self.dataframe)
                    plt.show()
                    DataFrame_Analyze(self.dataframe).anova_test(function,i,j)
                    DataFrame_Analyze(self.dataframe).turkeysd_test(i,j,alpha,groupby = groupby)
                    print('*********************************************************************************')
            else:
                raise ValueError('Only support for 2 categories variable analysis')
            print('')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('')
#14------------------------------------------------------------------------------------------
    def turkeysd_test(self,category,continous,alpha,groupby): # group by = 0 hoặc 1
        for name,grouped_df in self.dataframe[[category[0],category[1],continous]].groupby(category[groupby]):
            grouped_df = grouped_df.dropna(axis = 0)
            condition = grouped_df.value_counts().reset_index().groupby([category[0],category[1]]).agg(sum).reset_index()[0].to_list()
            if condition != []: 
                c = min(condition)
            else :
                c = 0
            if len(grouped_df) > 3 and len(grouped_df[category[1-groupby]].value_counts())>1 and c > 3:
                print('Hậu kiểm turkey')
                ICD.display(grouped_df)
                print(name,pairwise_tukeyhsd(grouped_df[continous],grouped_df[category[1 - groupby]],alpha = alpha))
            else :
                pass
#15------------------------------------------------------------------------------------------   
    def analyze_continous_vs_continous(self,type = 'heatmap'):
        ICD.display(self.dataframe.corr())
        plt.figure(figsize = (18,10))
        if type == 'heatmap':
            sb.heatmap(self.dataframe.corr(),annot = True,fmt='.2f',mask = np.triu(self.dataframe.corr()))
        else :
            sb.pairplot(self.dataframe)
        plt.show()
        
        
#16--------------------------------------------------------------------------------------------
    def in_out_category_category(self,in_variable,out_variable,alpha = 0.05):
        for i in in_variable :
            table = pd.crosstab(self.dataframe[i],self.dataframe[out_variable])
            print(f'Biến [{i}] và biến [{out_variable}]:')
            ICD.display(table)
            if DataFrame_Analyze(table).chi_2_test(alpha)[0] == True:
                print('Hai biến độc lập')
            else :
                print('Hai biến phụ thuộc nhau')
            table.plot.bar(stacked=True)
            plt.show()
            print('*****************************************************************************************')
        return
#17---------------------------------------------------------------------------------------------
    def in_out_continous_continous(self,in_variable,out_variable):
        for i in in_variable:
            print(f'Biến in : {i} và out : {out_variable}')
            a = self.dataframe[[i,out_variable]].corr()
            ICD.display(a)
            sb.regplot(x = self.dataframe[i], y = self.dataframe[out_variable])
            plt.show()
            print('******************************************************************************************')
        return
#18-----------------------------------------------------------------------------------------------
    def in_out_continous_category(self,in_variable,out_variable,alpha = 0.05,groupby = 0): #Đơn biến
        DataFrame_Analyze(self.dataframe).analyze_continous_vs_categories(in_variable,out_variable,alpha,groupby) 
        return
#19------------------------------------------------------------------------------------------------
    def in_out_category_continous(self,in_variable,out_variable,alpha = 0.05,groupby = 0):
        DataFrame_Analyze(self.dataframe).analyze_continous_vs_categories(in_variable,out_variable,alpha,groupby)
        return
#20-------------------------------------------------------------------------------------------------
    def table_category_vs_category(self,alpha = 0.05):
        var_1 = []
        var_2 = []
        chi_test = []
        for i in DataFrame_Analyze(self.dataframe).tw_variable():
            table = pd.crosstab(self.dataframe[i[0]],self.dataframe[i[1]])
            if len(table) != 0 :
                var_1.append(i[0])
                var_2.append(i[1])
                if DataFrame_Analyze(table).chi_2_test(alpha)[0] == True:
                    chi_test.append('Hai biến độc lập')
                else :
                    chi_test.append('Hai biến không độc lập')
        return(pd.DataFrame({'var_1':var_1,'var_2':var_2,'chi_test':chi_test}))



#21--------------------------------------------------------------------------------------------------------------
    def table_continous_vs_categories(self,category_vars_name,continous_vars_name,alpha = 0.05,groupby = 0):
        def match(data):
            n = len(data)
            _list = []
            for i in range(n):
                for j in range(i+1,n):
                    _list.append([data[i],data[j]])
            return _list
                #Chạy anova :
        a = []
        for j in continous_vars_name :
            if len(category_vars_name) == 1:
                function = f'{j} ~ C({category_vars_name[0]})'
                table_1 = pd.concat([pd.DataFrame({'continous':[j,j]}),DataFrame_Analyze(self.dataframe).anova_test2(function,category_vars_name,j).reset_index()],axis = 1) 
                a.append(table_1)   
            elif len(category_vars_name) > 1  :
                for i in match(category_vars_name) :
                    function = f"{j} ~ C({i[groupby]}) + C({i[1-groupby]}) + C({i[groupby]}):C({i[1-groupby]})"
                    table_1 = pd.concat([pd.DataFrame({'continous':[j,j,j,j]}),DataFrame_Analyze(self.dataframe).anova_test2(function,i,j).reset_index()],axis = 1)
                    a.append(table_1)
            else:
                raise ValueError('Only support for 2 categories variable analysis')
        return a
                
#22-------------------------------------------------------------------------------------------------------------
    def anova_test2(self,function,category,continous):
        if len(category) > 1 :
            model = ols(function, data=self.dataframe[[category[0],category[1],continous]]).fit()
            anova_table = sm.stats.anova_lm(model, type=2)
        else : 
            model = ols(function, data=self.dataframe[[category[0],continous]]).fit()
            anova_table = sm.stats.anova_lm(model, type=2)
        return anova_table 
#23---------------------------------------------------------------------------------------------------
    def get_outlier(self):
        q1 = np.quantile(self.dataframe,0.25)
        q3 = np.quantile(self.dataframe,0.75)
        iqr = q3-q1
        h_limit = q3 + iqr * 1.5  
        l_limit = q1 - iqr * 1.5
        return self.dataframe[(self.dataframe >= h_limit)|(self.dataframe <= l_limit)]
#24---------------------------------------------------------------------------------------------------
    def remove_outlier(self):
        q1 = np.quantile(self.dataframe,0.25)
        q3 = np.quantile(self.dataframe,0.75)
        iqr = q3-q1
        h_limit = q3 + iqr * 1.5  
        l_limit = q1 - iqr * 1.5
        return self.dataframe[(self.dataframe < h_limit)&(self.dataframe > l_limit)]
#25----------------------------------------------------------------------------------------------------
    def find_index_outlier(self):
        q1 = np.quantile(self.dataframe,0.25)
        q3 = np.quantile(self.dataframe,0.75)
        iqr = q3-q1
        h_limit = q3 + iqr * 1.5  
        l_limit = q1 - iqr * 1.5
        return self.dataframe.index[(self.dataframe >= h_limit)|(self.dataframe <= l_limit)]
#26------------------------------------------------------------------------------------------------------
    def get_unique_value_each_column(self):
        all_col = self.dataframe.columns
        for i in all_col :
            n = self.dataframe[i].unique()
            print(f'Biến {i} có {len(n)} giá trị :')
            print(n)
            count_ = self.dataframe[i].value_counts()
            perce_ = self.dataframe[i].value_counts(normalize = True)
            df = pd.concat([count_,perce_],axis = 1)
            df.columns = ['count','percent']
            DataFrame_Analyze(df).show()
#27----------------------------------------------------------------------------------------------------------
    def table_overall_unique_value_each_column(self):
        all_col = self.dataframe.columns
        len_lst = []
        value_lst = []
        for i in all_col:
            n = self.dataframe[i].unique()
            len_n = len(n)
            len_lst.append(len_n)
            value_lst.append(n)
        return pd.DataFrame({'feature':all_col,'value':value_lst,'number of unique value':len_lst})
#28------------------------------------------------------------------------------------------------------------
  
            
        
        









    
    

        
            
    
        
            
        


                
        
    
        
        
        
        
            
        
