# space allocating version1

假設有5個object、並將其分類為3種type，每個物件都有長、寬、高以及類型4種資訊。

程式概念參考三維裝箱問題常用之啟發式算法，不同之處在於普遍算法著重於空間利用率，也就是追求"整齊"，而這邊著重於如何疊的"合理"

為了儘量達到合理，會先將物件照著類型進行排序，並且將在各類型的物件中再以體積大小做排序

## pseudocode

    for i in object：
    
      第一個物件放置在座標（0, 0, 0）
      
      if 相同類型 & 往上放或往前放後之總長度不超過container ：
        
          將 i 放置之座標紀錄下來
          
        else：
        
          將其標記為放不進去的物件
          
      if 不同類型：
      
        先考慮往旁邊放，如果旁邊沒空間放才往不同類型的地方找


## problem

1.空間利用效率差

2.遇到特殊案例（ex:特別長or特別寬）應對能力差

# space allocating version2
1.新增視覺化後的堆疊樣貌

2.DP方法實現空間堆疊(先從只能放一層開始做起)

## problem
1.version1視覺化後可以發現有蠻多無視物理規則的情況

2.DP做出來會有overlap的情形


# space allocating version3
1.改用greedy作法，解決overlap問題，速度及擺放效果比DP好(參考2D_bin_packing.py)

2.將2D做法修改並套用到3D上(參考3D_bin_packing.py)

## problem
1.擺放邏輯可以有改進空間

2.未考慮物件堆疊的部分

# space allocating version4
1.增加物件堆疊的部分

2.嘗試引入cv2做物件偵測

## problem
1.堆疊邏輯的好壞有待公評(參考3D_bin_packing_V2.py)

3.物件偵測效果差 + 從照片無法得知三維訊息，就我查到的資料來看，似乎要搭配特殊的相機或是影片以及搭配一些AI輔助才有辦法(?(參考detect_obj.ipynb)

# space allocating version5
使用三種做法掃描物件並比較其優劣 :

	1.使用深度鏡頭獲得之深度圖做輪廓掃描(參考kinect2.py)

	2.深度鏡頭 + point cloud + DBSCAN(參考pointCloud_detect.py)

	3.一般鏡頭 + mediapipe套件(參考mediapipe_detect.py)

 ## problem
 做法1+做法2:
 
    電腦效能不足，在使用open3D套件時無法及時將每一幀畫面算出來

    Kinect生成的depth image雜訊偏多，容易掃描到非物件的東西

    掃描到物件後還需要再搭配模型去辨認物體類別
做法3:

    只能針對特定幾種物件做掃描


    


