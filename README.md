# DSAI_HW3  
## 題目  
綠能媒合交易平台，總金額越少越好    

## 資料  
過去七天資料(consumption、generation、bidresult)    

## 資料分析  
![](https://i.imgur.com/dk7y4YX.png)  
圖為範例資料前七天的用電量，可以看出高峰約在早上3至5點與10至20點    

![](https://i.imgur.com/A7AxTyV.png)
圖為範例資料前七天的產電量，可看見0點至7點與19點至24點都趨近於0    

  


## 方法  
由上圖可以判斷產電量跟用電量的差別    

使用LSTM預測未來一天(24小時)的產電量與用電量。  
訓練模型時，使用前四天(96筆)的資料當作輸入值，輸出值為24筆資料。    
將預測的結果將產電量與用電量相減，如果大於零代表有多的電可以賣，小於零則是需要買電，價格決定為買為2元，賣為2元和2.6元，量都為1    



## 計算效能
![](https://i.imgur.com/4vCNgXD.png)

B~h~ : Total power brought from platform (kWh) in the h-th hour
S~h~ : Total power sold from platform (kWh) in the h-th hour 
G~h~ : Total power generation (kWh) in the h-th hour 
C~h~ : Total power consumption (kWh) in the h-th hour 
trade_priceh : Bidding price in the h-th hour

## 實驗結果  
 
預測最後一天用電量:  
![](https://i.imgur.com/d2vtvzR.png)    
預測最後一天產電量:  
![](https://i.imgur.com/ajVs3N5.png)    
