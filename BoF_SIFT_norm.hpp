#include "prmu.hpp"

// include OpenCV libs
//#include <opencv2/opencv.hpp>
#include <set>
#include <random>
#include <algorithm>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>

using namespace std;
using namespace cv;

void BoF_SIFT(
	//
	// output
	prmu::ImageList (&imlist_result)[3], // 結果情報の記録用
	//
	// input
	size_t lv,
	const prmu::ImageList& imlist_learn,
	const prmu::ImageList (&imlist_test)[3]
)
{
	prmu::ImageList::const_iterator ite_learn, ite_test;
	prmu::ImageList::iterator ite_result;

    //make kd-tree
    vector <vector <float> > features;
    vector <int> partition;

    cv::SiftFeatureDetector detector;
    cv::SiftDescriptorExtractor extractor;

    map <prmu::Label,int> label_num;
    map <int,prmu::Label> trans_label;

    partition.push_back(0);

    //set <int> bad={7,17,18,23,25,26,28,30,33,34,36,50,51,52,56,57,58,60,61,62,65,66,71,72,73,77,79,83,84,85,87,96,100,108,109,112,113,119,121,123,124,125,128,129,130,131,134,135,136,139,141,147,149,150,154,155,157,158,162,167,171,173,177,179};

    int pos=0;
    for(ite_learn=imlist_learn.begin();ite_learn != imlist_learn.end();ite_learn++,pos++){
        //if(bad.count(pos)==1) continue;

        Mat learn=imread(ite_learn->full_file_path());

        vector <cv::KeyPoint> keypoints;
        Mat descriptor;
        detector.detect(learn, keypoints );
        /*for(int i=0;i<min(100,(int)keypoints.size());i++){
            circle(learn, keypoints[i].pt,1,Scalar(0,0,255));
        }
        imshow("shift_keypoints",learn);
        waitKey(0);*/

        extractor.compute(learn, keypoints, descriptor);

        label_num[ite_learn->label_of_1st_img()]+=descriptor.rows;
        //label_num[ite_learn->label_of_1st_img()]++;
        trans_label[trans_label.size()]=ite_learn->label_of_1st_img();

        for(int i=0;i<descriptor.rows;i++){
            vector <float> tmp;
            for(int j=0;j<descriptor.cols;j++){
                tmp.emplace_back(descriptor.at<float>(i,j));
            }
            
            float norm=sqrt(inner_product(tmp.begin(),tmp.end(),tmp.begin(),0.0));
                        
            for(int j=0;j<tmp.size();j++){
                    tmp[j]/=norm;
            }

            features.emplace_back(tmp);
        }

        partition.push_back(partition.back()+descriptor.rows);
    }

    cv::Mat features_mat(features.size(),128,CV_32F);
    for(int i=0;i<features.size();i++){
        for(int j=0;j<features[i].size();j++){
            features_mat.at<float>(i,j)=features[i][j];
        }
    }


    flann::Index kd(features_mat,flann::KDTreeIndexParams(100));
    //flann::Index kd(features_mat,flann::AutotunedIndexParams());

    cerr<<"successfully make kd-tree!"<<endl;

	for ( size_t _lv = 0; _lv < lv; ++_lv )
	{
		// アノテーション情報へのポインタ（イテレータ）
		// The pointer (actually iterator) of annotation information
		ite_test = imlist_test[_lv].begin();     // 入力用
		ite_result = imlist_result[_lv].begin(); // 結果用

		// アノテーション情報にアクセスするには，(*ite_test).XXX や ite_test->XXX が使用できます．
		// The access to the annotation information (i.e., member variables and functions)
		// is performed by (*ite_test).XXX or ite_test->XXX
        int tm=0;
        for(ite_test=imlist_test[_lv].begin();ite_test!=imlist_test[_lv].end();++ite_test,++ite_result,++tm){
            if(tm%20==0) cerr<<tm<<endl;

            vector <cv::KeyPoint> keypoints;
            Mat descriptor;

            cv::Mat test=imread(ite_test->full_file_path());
            size_t sy=test.rows, sx=test.cols; // size of image
            detector.detect(test, keypoints );

            extractor.compute(test, keypoints, descriptor);

            int row_size=descriptor.rows;
            int col_size=descriptor.cols;

            const int k=1;
            vector <int> indices(k);
            vector <float> dist(k);
            map <prmu::Label,int> points;

            for(int i=0;i<row_size;i++){
                /*const float *p=descriptor_float.ptr<float>(i);
                vector <float> query(p,p+col_size);*/
                vector <float> query(128);
                for(int j=0;j<128;j++){
                    query[j]=descriptor.at<float>(i,j);
                }
                
                float norm=sqrt(inner_product(query.begin(), query.end(), query.begin(),0.0));
                                                
                for(int j=0;j<128;j++){
                    query[j]/=norm;
                }

                kd.knnSearch(query, indices, dist, k, flann::SearchParams(100));
                //cerr<<dist[0]<<' '<<indices[0]<<endl;

                const float threthold=25000;
                prmu::Label label;
                for(int j=0;j<k;j++){
                    if(dist[j]<threthold){
                        int position=upper_bound(partition.begin(),partition.end(),indices[j])-partition.begin()-1;
                        label=trans_label[position];
                    }else{
                        label=prmu::LABEL_UNKNOWN;
                    }
                    if(label!=prmu::LABEL_UNKNOWN) points[label]++;
                }
            }

            prmu::Label candidate;
            double mx=0;
            for(auto it:points){
                cerr<<"Label number:"<<it.first<<", count="<<1.0*it.second/label_num[it.first]<<endl;
                if(mx<1.0*it.second/label_num[it.first]){
                    mx=1.0*it.second/label_num[it.first];
                    candidate=it.first;
                }
            }

            cerr<<"true label="<<ite_test->label_of_1st_img()<<", detection label="<<candidate<<endl;

            prmu::Rect bbox;

            bbox.x = sx/2;
            bbox.y = sy/2;
            bbox.w = sx/2;
            bbox.h = sy/2;
                // このデモでは，前処理において読み込んだ画像を 1/2 倍のサイズへと縮小したため，
                // もとのサイズの座標と対応させるため 2 倍している

            bbox = bbox.overlap( prmu::Rect(0,0,sx,sy) ); // 外接矩形が画像端をはみ出さないように補正（重要，スコアに影響する）

            ite_result->append_result( candidate, bbox );
        }
	} // end of the for-loop for levels
}
