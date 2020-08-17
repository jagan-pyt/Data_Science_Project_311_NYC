# Data_Science_Project_311_NYC
Data science project using python for Kaggle 311_NYC.

================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sklearn


service =pd.read_csv("D:\\Python\\Data\\311_Service.csv")

service.describe()

service.head()
service.tail()
service['Created Date']

service.info()

#checking shape and size
print(service.shape)
print(service.size)
service.head()

#Checking Null values in the Data
service.isnull().sum()

#Checking for unique values in each column
print(service['Complaint Type'].unique())
print(service['Descriptor'].unique())
print(service['City'].unique())

 'Astoria' 'Long Island City' 'Woodside' 'East Elmhurst' 'Howard Beach']
service.info()
service.dtypes

#CreatedDate and Closed date are Dtype - Object, make a note .

service
service.isnull().sum()

#Removing columns which has null values ( which does not show any impact on data)
service = service.drop(["Vehicle Type","Taxi Company Borough","Garage Lot Name","Ferry Direction","Ferry Terminal Name","Bridge Highway Segment","Taxi Pick Up Location"],axis=1)
service =service.drop("School or Citywide Complaint",axis=1)
service =service.drop(["Location","Longitude","Latitude","Road Ramp","Bridge Highway Direction","Y Coordinate (State Plane)","X Coordinate (State Plane)","Landmark","Cross Street 1","Cross Street 2","Intersection Street 1","Street Name","Incident Address","Intersection Street 2"],axis=1)

#print(service)
service.isnull().sum()
service.isna()
service['Descriptor'].value_counts()
service.columns

#Again check the Dataframe for analysis
service.head()
#service.info()
#we ca group few or major columns which are reason and which provoke to make a complaint and check their size
##print(service)

selected_columns = service[["Complaint Type","Descriptor"]]

#Creating a new Dataframe with required columns

Complaint_Attributes = selected_columns .copy()
Complaint_Attributes

service.groupby(['Complaint Type','Descriptor']).size()

#converting the columns ‘Created Date’ and Closed Date’ to datetime datatype and create a new column ‘Request_Closing_Time’ by parshing open and close dates and datetime package
import datetime
import time

service['Created Date']= pd.to_datetime(service['Created Date'])
service['Closed Date']= pd.to_datetime(service['Closed Date'])

dt = pd.DataFrame(data=service ,columns = ['Created Date','Closed Date'])
dt.dtypes

#We will check the average time to closed the case for that we need to know details of timings of creation and closing of the cases

service['Request_Closing_Time'] = service['Closed Date']-service['Created Date']


service['Request_Closing_Time'].describe()

#We can convert the above into data frame and also can replace NA values with dummies / mean value for precision however not required now

#Request_Closing_Time = pd.DataFrame(Request_Closing_Time,columns=['Case_closure_Time'])
#Request_Closing_Time

#Below is description of above result.Below we could see Average time taken to solve the case is  "04:18:51.832782"

#(Case_closure_Time count	298534 
#mean	0 days 04:18:51.832782
#std	0 days 06:05:22.141833
#min	0 days 00:01:00 
#25%	0 days 01:16:33 
#50%	0 days 02:42:55.500000 
#75%	0 days 05:21:00 
#max	24 days 16:52:22)###

#will check for nan missing closed /created date features
#Request_Closing_Time.isnull().sum()

#2164

#service_Request_Closing_Time.mean()

#service_Request_Closing_Time.fillna(service_Request_Closing_Time.mean(),inplace=True)

#service_Request_Closing_Time['Case_closure_Time'].isna()

#service_Request_Closing_Time
#From above we could see there are 2164 Null values in closed Date , since the case is pending we will omit that or we can replce Nat with Mean value for performance 

print(service['Request_Closing_Time'])
print(service['Request_Closing_Time'].dtypes)
0        00:55:15
1        01:26:16
2        04:51:31
3        07:45:14
4        03:27:02
           ...   
300693        NaT
300694   02:00:31
300695   03:07:17
300696   04:05:33
300697   04:08:49
Name: Request_Closing_Time, Length: 300698, dtype: timedelta64[ns]
timedelta64[ns]
Provide major insights/patterns that you can offer in a visual format (graphs or tables); at least 4 major conclusions that you can come up with after generic data mining.
DATA Mining

Let us consider the 4 conclusions in 4 aspects .

1) Complaint Type and Descriptor and their counts .

2) Differentiating the closure of cases depending on the time they took to close

3) Checking which Agency has worked better to clear / close the case asap

4) Status of complaints closed , pending etc ., with different features

Insight -1 , Complaint Type and Descriptor and their counts

print(service['Complaint Type'].unique())
print(service['Complaint Type'].value_counts())
plt.figure(figsize=(20,15))
service['Complaint Type'].value_counts().plot(kind='barh',color=list('rgbkymc'),alpha=0.75)
plt.title("Density of differnet non-emergency services")
plt.show
#This shows the occurence of more complaints in different occasions


plt.figure(figsize=(20,15))
service['Descriptor'].value_counts().plot(kind='barh',color=list('rgbkymc'),alpha=0.75)
plt.title("")
plt.show

2) Will categorize the response of agency based on time taken to resolve the case

Below 2 hours - Fast, Between 2 to 4 hours - Acceptable, Between 4 to 6 - Slow, More than 6 hours - Very Slow
Also please note that as we have subtracted one datetime from another so it gives you a timedelta object.
since our datetime is in days format let u convert it into hours
def toHr(X):
    days = X.days
    hours = round(X.seconds/3600, 2)
    result = (days * 24) + hours
    #print(days)
    #print(hours)
    return result
#change the Request closing time into Hours 

service['Request_Closing_Time_in_Hr']=service['Request_Closing_Time'].apply(toHr)

service['Request_Closing_Time_in_Hr'].dtypes
dtype('float64')
test_days = service[service['Unique Key'] == 30281825]['Request_Closing_Time']
test_days
300697   04:08:49
Name: Request_Closing_Time, dtype: timedelta64[ns]
#let us define a function for categorizing the case handling time  which was changed to Hour 

import math

def hourToCategory(hr):
    if (math.isnan(hr)):
        return 'Unspecified'
    elif (hr < 2.0):
        return 'Fast'
    elif (4.0 > hr >= 2.0):
        return 'Acceptable'
    elif (6.0 > hr >= 4.0):
        return 'Slow'
    else:
        return 'Very Slow'
#Need to categorize the request_closing_time column , for that we need to pass the Request_closing_time column to above functions to change the requesting time to hours and then caategorize.

service['Handling_Time_Category'] = service['Request_Closing_Time_in_Hr'].apply(hourToCategory)

service['Handling_Time_Category']
0                Fast
1                Fast
2                Slow
3           Very Slow
4          Acceptable
             ...     
300693    Unspecified
300694     Acceptable
300695     Acceptable
300696           Slow
300697           Slow
Name: Handling_Time_Category, Length: 300698, dtype: object
sns.countplot(service['Handling_Time_Category'])


sns.scatterplot(data=service,x='Handling_Time_Category',hue='City',y='Location Type')

sns.scatterplot(data=service,x='Handling_Time_Category',y='Location Type')

3 . Checking which Agency has worked better to clear / close the case asap
print(service['Agency Name'].value_counts())
print(service['Agency'].value_counts())
New York City Police Department    300690
Internal Affairs Bureau                 6
NYPD                                    2
Name: Agency Name, dtype: int64
NYPD    300698
Name: Agency, dtype: int64
We could have checked which agency has resoponded well but since all comes under same agency believe it does not make much differnece so we can conclude agency has the same work on all complaint types
4) Status of complaints closed , pending etc ., with different features
#As per provided data there are many columns with unneccesary data which needs to be removed or need to create a new data frame with required features
#services_emergency = service.groupby(['Complaint Type','Descriptor','City','Borough','Status','Created Date','Closed Date'])
#services_emergency.dtypes
Services_emergency = pd.DataFrame({'count':
                                   service.groupby(['Complaint Type','Status',]).size()})
print(Services_emergency)
Services_emergency.info()
Services_emergency.isna()
#Services_emergency.describe()
Services_emergency.head()
#Services_emergency["Complaint Type"].unique()
Services_emergency.tail()
                                    count
Complaint Type            Status         
Agency Issues             Closed        6
Animal Abuse              Assigned      4
                          Closed     7766
                          Open          8
Animal in a Park          Closed        1
Bike/Roller/Skate Chronic Assigned      1
                          Closed      424
                          Open          2
Blocked Driveway          Assigned     98
                          Closed    76793
                          Draft         1
                          Open        152
Derelict Vehicle          Assigned     27
                          Closed    17585
                          Open        106
Disorderly Youth          Closed      286
Drinking                  Closed     1275
                          Open          5
Ferry Complaint           Open          2
Graffiti                  Closed      113
Homeless Encampment       Assigned      4
                          Closed     4410
                          Open          2
Illegal Fireworks         Closed      168
Illegal Parking           Assigned    257
                          Closed    74515
                          Open        589
Noise - Commercial        Assigned    156
                          Closed    35245
                          Open        176
Noise - House of Worship  Closed      929
                          Open          2
Noise - Park              Assigned     11
                          Closed     4021
                          Open         10
Noise - Street/Sidewalk   Assigned    201
                          Closed    48068
                          Draft         1
                          Open        342
Noise - Vehicle           Assigned     20
                          Closed    17032
                          Open         31
Panhandling               Assigned      1
                          Closed      305
                          Open          1
Posting Advertisement     Assigned      1
                          Closed      647
                          Open          2
Squeegee                  Closed        4
Traffic                   Assigned      2
                          Closed     4493
                          Open          3
Urinating in Public       Closed      592
Vending                   Assigned      3
                          Closed     3793
                          Open          6
<class 'pandas.core.frame.DataFrame'>
MultiIndex: 56 entries, ('Agency Issues', 'Closed') to ('Vending', 'Open')
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   count   56 non-null     int64
dtypes: int64(1)
memory usage: 902.0+ bytes
count
Complaint Type	Status	
Traffic	Open	3
Urinating in Public	Closed	592
Vending	Assigned	3
Closed	3793
Open	6
sns.scatterplot(x=service['Status'],y=service['Location Type'])

plt.figure(figsize=(10,8))
service['Status'].hist(bins=60)
plt.title("Status of cases")
plt.xlabel("Status_Category")
plt.ylabel("No.of cases")
plt.show()

# 4 . Order the complaint types based on the average ‘Request_Closing_Time’, grouping them for different locations.
#For the above we will bring back our previous finding 'Request_Closing_Time' and find the mean of that, also please check for null values in each column

service['Request_Closing_Time_in_Hr']
service['Request_Closing_Time_in_Hr'].isnull().sum()
print(service['City'].isnull().sum())
print(service['Complaint Type'].isnull().sum())
# we could see there are no null/na values in city and complaint type except in Request closing time.
2614
0
service['Request_Closing_Time'].mean()
Timedelta('0 days 04:18:51.832782')
#we will create a column with mean of Request closing time and  replace NA with mean.

service['Mean_Request_Closing_Time_in_Hr'] = service['Request_Closing_Time_in_Hr'].fillna(service['Request_Closing_Time'].mean())
service['Mean_Request_Closing_Time_in_Hr']
0                           0.92
1                           1.44
2                           4.86
3                           7.75
4                           3.45
                   ...          
300693    0 days 04:18:51.832782
300694                      2.01
300695                      3.12
300696                      4.09
300697                      4.15
Name: Mean_Request_Closing_Time_in_Hr, Length: 300698, dtype: object
service['Request_Closing_Time_in_Hr'].dtypes
dtype('float64')
select_col = service[["City","Complaint Type","Request_Closing_Time_in_Hr"]]
Service_sorting_with_Request_time = select_col.copy()
Service_sorting_with_Request_time.sort_values("Request_Closing_Time_in_Hr",axis= 0, ascending = True,na_position ='last').dropna()
City	Complaint Type	Request_Closing_Time_in_Hr
61125	NEW YORK	Noise - Commercial	0.02
163683	BROOKLYN	Noise - Commercial	0.03
222119	STATEN ISLAND	Noise - Street/Sidewalk	0.03
60357	STATEN ISLAND	Posting Advertisement	0.03
260083	NEW YORK	Noise - Commercial	0.03
...	...	...	...
12168	BROOKLYN	Derelict Vehicle	223.37
283132	QUEENS	Animal in a Park	336.83
21268	BROOKLYN	Animal Abuse	519.27
23664	BROOKLYN	Illegal Parking	577.36
244488	BROOKLYN	Noise - Street/Sidewalk	592.87
298028 rows × 3 columns

service_group_by_city_Request_time = pd.DataFrame(service.groupby(['City','Request_Closing_Time_in_Hr']).size())
service_group_by_city_Request_time
0
City	Request_Closing_Time_in_Hr	
ARVERNE	0.20	1
0.22	1
0.24	1
0.30	1
0.31	1
...	...	...
Woodside	13.71	1
15.57	1
16.13	1
16.93	1
28.17	1
40799 rows × 1 columns

service_group_by_city_Request_time.loc['BROOKLYN']
#service_group_by_city_Request_time.loc['BROOKLYN'].sum()
0
Request_Closing_Time_in_Hr	
0.03	1
0.04	1
0.05	15
0.06	20
0.07	72
...	...
223.35	1
223.37	1
519.27	1
577.36	1
592.87	1
2855 rows × 1 columns

select_columns = service[["City","Complaint Type","Request_Closing_Time"]]
service['City'].isnull().sum()
2614
sns.boxplot(y=service['City'],x=service['Request_Closing_Time_in_Hr'])

service_data_mined = select_columns .copy()

service_data_mined['Request_Closing_Time'].fillna(service['Request_Closing_Time'].mean(),inplace=True)
service_data_mined['Request_Closing_Time_in_hr']=service_data_mined['Request_Closing_Time'].apply(toHr)

service_data_mined
City	Complaint Type	Request_Closing_Time	Request_Closing_Time_in_hr
0	NEW YORK	Noise - Street/Sidewalk	00:55:15	0.92
1	ASTORIA	Blocked Driveway	01:26:16	1.44
2	BRONX	Blocked Driveway	04:51:31	4.86
3	BRONX	Illegal Parking	07:45:14	7.75
4	ELMHURST	Illegal Parking	03:27:02	3.45
...	...	...	...	...
300693	NaN	Noise - Commercial	04:18:51.832782	4.31
300694	RICHMOND HILL	Blocked Driveway	02:00:31	2.01
300695	BROOKLYN	Noise - Commercial	03:07:17	3.12
300696	BRONX	Noise - Commercial	04:05:33	4.09
300697	NEW YORK	Noise - Commercial	04:08:49	4.15
300698 rows × 4 columns

Hypothesis_Testing -- Anova Testing
Whether the average response time across complaint types is similar or not (overall)

# H0 : All Complain Types average response time mean is similar
#H1 : All Complain Types average response time mean is Not similar
# I choose a traditional method to do our calculatin and find P - value to prove the same .
#for that let us create a data frame where we have Complaint types and mean of each complaint type so that we can check for relation between them.
import statistics
import scipy
from scipy import stats
#For my convenience I am making two rows Com_Type: for complaint type and Average  for Request_closing_Time_in_Hr 
Hypothesis_testing= pd.DataFrame({'Com_Type':service['Complaint Type'],'Average':service_data_mined['Request_Closing_Time_in_hr']})
Hypothesis_testing.dropna()
Hypothesis_testing.isnull().sum()
                                  
Com_Type    0
Average     0
dtype: int64
#For my convenience I am making two rows Com_Type: for complaint type and Mean for Request-closing_tim_in_hour 
Hypothesis_testing['Mean'] = Hypothesis_testing.groupby('Com_Type')['Average'].transform(np.mean)
Hypothesis_testing.dropna()

#for our understanding I have shown you the sample average and Complaint_type average below which is necessary for calculating pvalue or doint anova test
Com_Type	Average	Mean
0	Noise - Street/Sidewalk	0.92	3.454757
1	Blocked Driveway	1.44	4.739595
2	Blocked Driveway	4.86	4.739595
3	Illegal Parking	7.75	4.499049
4	Illegal Parking	3.45	4.499049
...	...	...	...
300693	Noise - Commercial	4.31	3.157953
300694	Blocked Driveway	2.01	4.739595
300695	Noise - Commercial	3.12	3.157953
300696	Noise - Commercial	4.09	3.157953
300697	Noise - Commercial	4.15	3.157953
300698 rows × 3 columns

#Please have a look on our data with the below picture
service_data_mined.boxplot('Request_Closing_Time_in_hr', by='Complaint Type', figsize=(20,8))

#Create a data Frame with Complaint type and response time to carry out annova test on the below

sel_col = service[["Complaint Type","Mean_Request_Closing_Time_in_Hr"]]
Hypothesis_Test = sel_col.copy()
Hypothesis_Test
Hypothesis_Test.dropna()
Hypothesis_Test.isnull().sum()
Complaint Type                     0
Mean_Request_Closing_Time_in_Hr    0
dtype: int64
print(np.var(Hypothesis_testing['Average']))
36.81489375666578
##Hypothesis_testing= Hypothesis_testing[['Com_Type','Average']]
result = pd.unique(Hypothesis_testing.Com_Type.values)
result_data = {anova:Hypothesis_testing['Average'][Hypothesis_testing.Com_Type == anova] for anova in result}
result_data
F, p = stats.f_oneway(result_data['Noise - Street/Sidewalk'],result_data['Illegal Parking'],result_data['Noise - Commercial'],result_data['Blocked Driveway'],result_data['Animal Abuse'],result_data['Derelict Vehicle'])
print(F, p)
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
     print("accept null hypothesis")
        
#You can take any sample of data from complaint type there is no relationship at all... i have tried many combinations ..Infact a single combination would show you the result 
1423.930958657441 0.0
p-value for significance is:  0.0
reject null hypothesis
From above we can reject the null hypothesis which is :All Complain Types average response time mean are not similar

Are the type of complaint or service requested and location related?Are the type of complaint or service requested and location related?
# W e would like to check if Complint type and Serive are realted to location , for a minor knowledge you can see a chart above
#where you would get a idea of it.

#The chi-square test of independence can be used to examine this relationship. 
#H0:The null hypothesis for this test is that there is no relationship between TWO ENTITIES  
#H1:The alternative hypothesis is that there is a relationship between Two entities.
from scipy.stats import chi2
from scipy.stats import chi2_contingency
#For carrying ou our Chi-Square test let us first create a table with complaint_type counts and city (location).
Data_Complaint = service['Complaint Type'].value_counts()[:5]
Data_Complaint
Blocked Driveway           77044
Illegal Parking            75361
Noise - Street/Sidewalk    48612
Noise - Commercial         35577
Derelict Vehicle           17718
Name: Complaint Type, dtype: int64
Data_Complaint_Major = Data_Complaint.index
Data_Complaint_Major
Index(['Blocked Driveway', 'Illegal Parking', 'Noise - Street/Sidewalk',
       'Noise - Commercial', 'Derelict Vehicle'],
      dtype='object')
Data_City = service['City'].value_counts()[:5]
Data_City

# The beloe are the top 5 location types with many complaints 
BROOKLYN         98307
NEW YORK         65994
BRONX            40702
STATEN ISLAND    12343
JAMAICA           7296
Name: City, dtype: int64
Major_complaints_location = Data_City.index
Major_complaints_location
Index(['BROOKLYN', 'NEW YORK', 'BRONX', 'STATEN ISLAND', 'JAMAICA'], dtype='object')
#Will make a matrix of the above two

Major_complaints_location_and_complaint_type = service.loc[(service['Complaint Type'].isin(Data_Complaint_Major)) & (service['City'].isin(Major_complaints_location)), ['Complaint Type', 'City']]

Major_complaints_location_and_complaint_type

#From above we could see Major complaint types and locations where the complaint has been raised  with indexes of those.
Complaint Type	City
0	Noise - Street/Sidewalk	NEW YORK
2	Blocked Driveway	BRONX
3	Illegal Parking	BRONX
5	Illegal Parking	BROOKLYN
6	Illegal Parking	NEW YORK
...	...	...
300691	Noise - Commercial	NEW YORK
300692	Noise - Commercial	NEW YORK
300695	Noise - Commercial	BROOKLYN
300696	Noise - Commercial	BRONX
300697	Noise - Commercial	NEW YORK
185475 rows × 2 columns

#The pandas crosstab function builds a cross-tabulation table that can show the frequency with which certain groups of data appear

Chi_Table=pd.crosstab(Major_complaints_location_and_complaint_type['Complaint Type'],Major_complaints_location_and_complaint_type['City'],margins=True)
#Applying chisquare technique on the above.
stats.chi2_contingency(Chi_Table)
(40522.79928349594,
 0.0,
 25,
 array([[  8759.46060116,  22125.43658175,   1540.58786899,
          12849.86224559,   2659.65270252,  47935.        ],
        [  1898.81203936,   4796.19091522,    333.95741205,
           2785.49950128,    576.54013209,  10391.        ],
        [  9823.16812508,  24812.24510042,   1727.6695835 ,
          14410.28882599,   2982.62836501,  53756.        ],
        [  5400.58618143,  13641.28825987,    949.83902952,
           7922.49564631,   1639.79088287,  29554.        ],
        [  8010.97305297,  20234.83914274,   1408.94610594,
          11751.85378083,   2432.38791751,  43839.        ],
        [ 33893.        ,  85610.        ,   5961.        ,
          49720.        ,  10291.        , 185475.        ]]))
stats.chi2_contingency
Returns

chi2float : The test statistic.

pfloat : The p-value of the test

dofint : Degrees of freedom

expectedndarray, same shape as observed.

The expected frequencies, based on the marginal sums of the table.

from above The P vale is 0.0 which is less than 0.05 which means we reject null hypothesis and can conclude there is a relationship between the location and Complaint type .
