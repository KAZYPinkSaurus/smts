import os
import pandas as pd
import numpy as np

#あるディレクトリ内のcsvファイルを全て変換
# input: aPath 変換したいデータのあるディレクトリのパス, aSidName sidとなるカラム名, aDropList 使用しないカラム名
def seqAddVec(aPath,aSidName,aDropList):
    # @t,x(t),x(t)-x(t-1),....と変換 return
    #時間をindexに変換
    #x_1(t)-x_1(t-1).....x_m(t)-x_m(t-1)を追加
    tFileLists = os.listdir(aPath)
    tOutputFrame = pd.DataFrame()
    for i in tFileLists:
        df  = pd.read_csv(aPath+i)
        
        tIndex =pd.DataFrame(np.array(range(len(df))),columns=["t"])
        #使わないカラム削除
        for i in aDropList:
            df = df.drop(i,axis=1)

        id = df[aSidName]
        df = df.drop(aSidName,axis=1)
        
        diff = pd.DataFrame(df[1:len(df.index)]-df[0:len(df.index)-1].values)
        
        diff = diff.rename(columns=lambda s:s+"_diff")
        df =pd.concat([df,diff],axis=1)
        df =pd.concat([tIndex,df],axis=1)
        df =pd.concat([id,df],axis=1)
    
        tOutputFrame =pd.concat([tOutputFrame,df],ignore_index=True)
    
    return tOutputFrame.fillna(0)

