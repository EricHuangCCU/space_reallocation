# space allocating version1

假設有5個object、並將其分類為3種type，每個物件都有長、寬、高以及類型4種資訊。

程式概念參考三維裝箱問題常用之啟發式算法，不同之處在於普遍算法著重於空間利用率，也就是追求"整齊"，而這邊著重於如何疊的"合理"
為了儘量達到合理，會先將物件照著類型進行排序，並且將在各類型的物件中再以體積大小做排序

# pseudocode

    for i in object：
    
      第一個物件放置在座標（0, 0, 0）
      
      if 相同類型：
      
        if往上放或往前放後之總長度不超過container ：
        
          將 i 放置之座標紀錄下來
          
        else：
        
          將其暫時紀錄下來
          
      if 不同類型：
      
        先考慮往旁邊放，如果旁邊沒空間放才往不同類型的地方找


# problem
