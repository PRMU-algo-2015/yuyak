#include "prmu.hpp"

// include OpenCV libs
//#include <opencv2/opencv.hpp>
#include <set>
#include <random>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>

using namespace std;
using namespace cv;

void SIFT_BoF_SVM(
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


    //set <int> bad={7,17,18,23,25,26,28,30,33,34,36,50,51,52,56,57,58,60,61,62,65,66,71,72,73,77,79,83,84,85,87,96,100,108,109,112,113,119,121,123,124,125,128,129,130,131,134,135,136,139,141,147,149,150,154,155,157,158,162,167,171,173,177,179};

    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);

    cv::Mat descriptors;
    vector <int> partition;

    partition.push_back(0);

    //gather SIFT features

    for (ite_learn = imlist_learn.begin(); ite_learn != imlist_learn.end(); ite_learn++) {
        Mat learn = imread(ite_learn->full_file_path(), 0);

        vector <cv::KeyPoint> keypoints;
        Mat descriptor;

        detector->detect(learn, keypoints );
        extractor->compute(learn, keypoints, descriptor);

        descriptors.push_back(descriptor);

        partition.push_back(partition.back() + descriptor.rows);
    }

    //make bag of features ::: kmeans clustering and convert to histgram

    const int cluster_num = 25;
    BOWKMeansTrainer bofTrainer(cluster_num, cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6), 1, KMEANS_PP_CENTERS);

    Mat dictionary = bofTrainer.cluster(descriptors);

    //preserve
    /*FileStorage fs("dictionary.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();*/


    //svm training
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    bowDE.setVocabulary(dictionary);

    Mat histgram;
    Mat learn_label;

    for (ite_learn = imlist_learn.begin(); ite_learn != imlist_learn.end(); ite_learn++) {
        Mat learn_img = imread(ite_learn->full_file_path(), 0);

        vector<KeyPoint> sift_keypoints;
        detector->detect(learn_img, sift_keypoints);

        Mat bowDescriptor;
        bowDE.compute(learn_img, sift_keypoints, bowDescriptor);

        histgram.push_back(bowDescriptor);
        learn_label.push_back((int)ite_learn->label_of_1st_img() / 10);
    }

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 500, 0.0001);
    params.C = 300;
    params.gamma = 0.5;

    CvSVM svm;

    if (svm.train_auto(histgram, learn_label, Mat(), Mat(), params)) {
        cerr << "successfully train!" << endl;
    } else {
        assert(0);
    }

    /*CvSVMParams now=svm.get_params();
    dump(now.C);
    dump(now.kernel_type);
    dump(now.nu);
    dump(now.gamma);
    dump(now.p);*/

    map <prmu::Label, pair<int, int> > answer_rate;

    for ( size_t _lv = 0; _lv < lv; ++_lv )
    {
        // アノテーション情報へのポインタ（イテレータ）
        // The pointer (actually iterator) of annotation information
        ite_test = imlist_test[_lv].begin();     // 入力用
        ite_result = imlist_result[_lv].begin(); // 結果用

        // アノテーション情報にアクセスするには，(*ite_test).XXX や ite_test->XXX が使用できます．
        // The access to the annotation information (i.e., member variables and functions)
        // is performed by (*ite_test).XXX or ite_test->XXX

        for (ite_test = imlist_test[_lv].begin(); ite_test != imlist_test[_lv].end(); ++ite_test, ++ite_result) {
            Mat test_img = imread(ite_test->full_file_path(), 0);
            size_t sy = test_img.rows, sx = test_img.cols; // size of image

            vector<KeyPoint> sift_keypoints;
            detector->detect(test_img, sift_keypoints);

            Mat test_bowDescriptor;
            bowDE.compute(test_img, sift_keypoints, test_bowDescriptor);

            cv::Mat result;

            svm.predict(test_bowDescriptor, result);

            int label = result.at<float>(0, 0);

            cerr << "true label=" << ite_test->label_of_1st_img() << ", detection label=" << label * 10 << endl;
            answer_rate[ite_test->label_of_1st_img()].first += prmu::label::issame(ite_test->label_of_1st_img(), (prmu::Label)(label * 10));
            answer_rate[ite_test->label_of_1st_img()].second++;

            prmu::Rect bbox;

            bbox.x = sx / 2;
            bbox.y = sy / 2;
            bbox.w = sx / 2;
            bbox.h = sy / 2;
            // このデモでは，前処理において読み込んだ画像を 1/2 倍のサイズへと縮小したため，
            // もとのサイズの座標と対応させるため 2 倍している

            bbox = bbox.overlap( prmu::Rect(0, 0, sx, sy) ); // 外接矩形が画像端をはみ出さないように補正（重要，スコアに影響する）

            ite_result->append_result( (prmu::Label)(label * 10), bbox );
        }
        for (auto it : answer_rate) {
            cerr << "PRMU::Label:" << it.first << ",　answer rate:" << 1.0 * it.second.first / it.second.second << endl;
        }
    } // end of the for-loop for level
}

