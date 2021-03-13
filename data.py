import pandas as pd
import numpy as np


if __name__ == "__main__":
  
  data = pd.read_csv('monojet_Zp2000.0_DM_50.0_chan3.csv',sep = ";", header=None,names=list(range(0,17)))
  ### Handling Columns
  d1 = {0:'event ID',1:'process ID',2:'event weight',3:'MET',4:'METphi'}
  dic = {}
  for i in range(5,17):
    dic[i] = 'obj_' + str(i)
    dic.update(d1)
  data.rename(columns=dic,inplace=True)
  ### filling null values with 0
  data.fillna(0,inplace=True)
  print(data.head(5))
  ### Storing the values in arr that contain "j" in it
  arr = []
  for j in range(12):
    for i in range(len(data)):
      a = data.loc[i,'obj_'+ str(j+5)]
      if a != 0:
        if a.split(",")[0] == 'j':
          arr.append([a.split(",")[1:]])

  #### creating dataframe with arr
  train_data = pd.DataFrame(arr)
  train_data.rename(columns={0:"KineticsFeatures"},inplace=True)

### Example
#                                           0
###0  [1.06946e+06, 751597, 0.858186, -1.84217]
###1        [676000, 640429, 0.33045, 0.704554]
###2       [936707, 616229, 0.973383, -1.56592]
###3        [640313, 589524, 0.390749, 1.23734]
###4       [583373, 545730, 0.364057, -1.60732]
  

  train_data.loc[:,'pt'] = train_data["KineticsFeatures"].apply(lambda x : float(x[1]))
  train_data.loc[:,'eta'] = train_data["KineticsFeatures"].apply(lambda x : float(x[2]))
  train_data.loc[:,'phi'] = train_data["KineticsFeatures"].apply(lambda x : float(x[3]))
  train_data.loc[:,'E'] = train_data["KineticsFeatures"].apply(lambda x : float(x[0]))
  ### Droping non relevant columns
  train_data.drop("KineticsFeatures",axis =1,inplace=True)

  print(train_data.head(5))
  train_data.to_pickle('jetData.pkl')


