PRMUアルゴリズムコンテスト2015に挑戦してみた

コンテスト概要はhttps://sites.google.com/site/alcon2015prmu/home

靴の画像群からメーカーのロゴを抽出する課題

現在の方針:
    SIFT特徴量を用いてBoF(Bag of features)を作成する. 二乗距離最小のものに投票を行う.

コード:  
    random_labeling.hpp : ラベルをランダムに割り振る   
    voting_SIFT.hpp : SIFT  
    voting_SIFT_norm.hpp : normalization SIFT  
    SIFT_BoF_SVM.hpp : SIFT+BoF+SVM  

結果:
    result.txt

発表資料:
    doc/17.pdf

