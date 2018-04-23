from sklearn.ensemble import RandomForestClassifier
import collections
import pandas as pd
import numpy as np

# trainを元に学習を行い, train,testともに変換を行う.R, Jins ,NTreeはパラメータ
#iput: trainのラベルy,変換後のデータX(tripIDをsidとしてつけておく),aRはシンボル数(2の倍数で), aJinsは最終的なtreeの数, aNTreeはRFを構成するtreeの数,
#output: sid + 変換後のtrain , sid + 変換後のtest
def smts(aTrainY,aTrainX,aTestX,aR, aJins,aNTree):

    tTrainOut = pd.DataFrame()
    tTestOut = pd.DataFrame()
    
    tTrainX = aTrainX.iloc[:,1:len(aTrainX.columns)]
    #trainのsidリスト
    tTrSids = pd.DataFrame(aTrainX[aTrainX.columns[0]],columns=[aTrainX.columns[0]])
    tTestX = aTestX.iloc[:,1:len(aTestX.columns)]
    #testのsidリスト
    tTeSids = pd.DataFrame(aTestX[aTestX.columns[0]],columns=[aTestX.columns[0]])

    # uniqueな状態になったsid
    sid_tr = pd.DataFrame()
    sid_te = pd.DataFrame()

    # 学習機RFinsをaJins個作成
    for i in range(aJins):

        clf = RandomForestClassifier(n_estimators=aNTree,max_leaf_nodes=int((aR+2)/2))


        #trainで学習
        clf.fit(tTrainX, aTrainY)
        
        #train testともにNode計算　
        tIdxTr = clf.apply(tTrainX)
        tIdxTe = clf.apply(tTestX)
        
        #変換完了(1つのTree) [N x R]
        sid_tr,tTrX = H_jX(tIdxTr,tTrSids,aR)
        sid_te,tTeX = H_jX(tIdxTe,tTeSids,aR)

        #outputに結合
        tTrainOut=pd.concat([tTrainOut,tTrX],axis=1,ignore_index=True)
        tTestOut =pd.concat([tTestOut,tTeX],axis=1,ignore_index=True)

        print("Done "+str(i+1)+"th RF")
    

    #TrainOutは[sid]+[N x RJins]
    #TestOutは[sid]+[N x RJins]
    #DataFrame型にする
    return pd.concat([sid_tr,tTrainOut],axis=1),pd.concat([sid_te,tTestOut],axis=1)

# RFで分類されたノードindexがsidごとにどんな割合で分布しているかの値に変換
# aLabels_jn:RFで分類された値リスト,aSids:sidリスト(dataframe),aR:RFでのノード数
def H_jX(aLabels_jn,aSids,aR):
    tSidName=aSids.columns[0]
    tOutX = pd.DataFrame()
    tSids = pd.DataFrame(columns=[tSidName])
    tSid = aSids.iloc[0,0]
    tCntIdx = np.zeros(aR)
    tCntSid = 0
    for i in range(len(aLabels_jn)):
        if(tSid == aSids.iloc[i][0]):
            tCntSid += 1
            # 一番多い,index
            tIdx = collections.Counter(aLabels_jn[i]).most_common()[0][0]
            tCntIdx[tIdx-1] += 1
            #最後
            if(i == len(aLabels_jn)-1):
                #正規化
                tCntIdx = tCntIdx/tCntSid
                #結合
                tOutX = pd.concat([tOutX,pd.DataFrame(tCntIdx).T],ignore_index=True)
                tSids= pd.concat([tSids,pd.DataFrame([tSid],columns=[tSidName])],ignore_index=True)
                
        else:
            #正規化
            tCntIdx = tCntIdx/tCntSid
            #結合
            tOutX = pd.concat([tOutX,pd.DataFrame(tCntIdx).T],ignore_index=True)
            tSids= pd.concat([tSids,pd.DataFrame([tSid],columns=[tSidName])],ignore_index=True)

            #次のsid
            tSid = aSids.iloc[i][0]

            tCntIdx = np.zeros(aR)
            tCntSid = 1

            tIdx = collections.Counter(aLabels_jn[i]).most_common()[0][0]
            tCntIdx[tIdx-1] += 1

    return tSids,tOutX
