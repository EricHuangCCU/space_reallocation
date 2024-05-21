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
1.改用greedy作法，解決overlap問題，速度及擺放效果比DP好

2.將2D做法修改並套用到3D上

## problem
1.擺放邏輯可以有改進空間

2.未考慮物件堆疊的部分
