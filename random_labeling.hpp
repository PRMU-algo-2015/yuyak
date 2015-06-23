#include <prmu.hpp>

#include <set>
#include <random>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void random_labeling(prmu::ImageList (&imlist_result)[3], size_t lv, const prmu::ImageList& imlist_learn, const prmu::ImageList (&imlist_test)[3])
{
    prmu::ImageList::const_iterator ite_learn, ite_test;
    prmu::ImageList::iterator ite_result;
    set <prmu::Label> random_label;
    for (ite_learn = imlist_learn.begin(); ite_learn != imlist_learn.end(); ite_learn++) {
        random_label.insert(ite_learn->label_of_1st_img());
    }

    for ( size_t _lv = 0; _lv < lv; ++_lv ) {
        ite_result = imlist_result[_lv].begin(); // 結果用
        for (ite_test = imlist_test[_lv].begin(); ite_test != imlist_test[_lv].end(); ++ite_test, ++ite_result) {
            Mat test = imread(ite_test->full_file_path());
            size_t sy = test.rows, sx = test.cols; // size of image

            prmu::Rect bbox;

            bbox.x = sx / 2;
            bbox.y = sy / 2;
            bbox.w = sx / 2;
            bbox.h = sy / 2;
            // このデモでは，前処理において読み込んだ画像を 1/2 倍のサイズへと縮小したため，
            // もとのサイズの座標と対応させるため 2 倍している

            bbox = bbox.overlap( prmu::Rect(0, 0, sx, sy) ); // 外接矩形が画像端をはみ出さないように補正（重要，スコアに影響する）
            uniform_int_distribution <size_t> uni(0, random_label.size() - 1);

            set <prmu::Label>::const_iterator it(random_label.begin());
            mt19937 mt(0);
            advance(it, uni(mt));
            prmu::Label lab = *it;
            ite_result->append_result(lab, bbox );
        }
    }
}