package utils;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Created by Paindar on 17/12/12.
 */
public class ImageUtils
{
    public static Mat add_overlay(Mat image, Mat overlay, double alpha, double beta)
    {
        int image_i = image.rows();
        int image_j = image.cols();
        int overlay_i = overlay.rows();
        int overlay_j = overlay.cols();
        Mat _dst = image.clone();

        UByteIndexer srcIndexer = image.createIndexer(),
                overlayIndexer = overlay.createIndexer(),
                dstIndexer = _dst.createIndexer();
        for(int i=0;i<image_i;i++){
            for(int j=0;j<image_j;j++){
                int values[]=new int[3];
                int ovr[] = new int[3];
                int res[] = new int[3];
                srcIndexer.get(i, j, values);
                overlayIndexer.get((i * overlay_i / image_i), (j * overlay_j / image_j), ovr);
                for(int k=0;k<3;k++){
                    res[k] = (int)(alpha * values[k] + beta * ovr[k]);
                }
                dstIndexer.put(i, j, res);
            }
        }
        return _dst;
    }

    public static Mat blur(Mat image, double alpha)
    {
        int _size = (int)(alpha*image.rows());
        System.out.println("size = "+_size);
        if(_size % 2 == 0){
            _size += 1;
        }
        Mat dst = image.clone();
        GaussianBlur(image,dst,new Size(_size, _size), 0);
        return dst;
    }

    public static Map<Long,int[]> init_color_table(String path)
    {
        Mat img = imread(path);
        Map<Long,int[]> color_table = new HashMap<>();
        if(img==null || img.rows()==0)
            return null;
        UByteIndexer indexer = img.createIndexer();
        for(int i=0;i<indexer.rows();i++){
            for(int j=0;j<indexer.cols();j++){
                int point[] = new int[3];
                indexer.get(i,j,point);
                long key = (((4*(point[0]/4))*256) + (4*(point[1]/4)))*256+ (4*(point[2]/4));
                if (color_table.get(key)==null){
                    color_table.put(key, new int[]{i,j});
                }
                else{
                    System.out.println("find existed key:"+ key);
                }
            }
        }
        return color_table;
    }

    public static Mat add_filter(Mat img, Map<Long, int[]> table, Mat flt)
    {
        Mat  _dst = img.clone();
        UByteIndexer indexer = img.createIndexer(),fltColor = flt.createIndexer(),dstIndex=_dst.createIndexer();
        for(int i=0;i<indexer.rows();i++)
        {
            for (int j = 0; j < indexer.cols(); j++)
            {
                int point[] = new int[3];
                indexer.get(i,j,point);
                long key = (((4*(point[0]/4))*256) + (4*(point[1]/4)))*256+ (4*(point[2]/4));
                int pos[] = table.get(key);
                int new_color[] = new int[3];
                fltColor.get(pos[0],pos[1],new_color);
                dstIndex.put(i,j,new_color);
            }
        }
        return _dst;
    }

    public static int[] gray_quantities(Mat img, int flat_min, int flat_max)
    {
        int p_min_pixel = 0;
        int p_max_pixel = 0;
        int h_size = 256;
        int histogram[] = new int[256];
        int len_size = img.rows();
        int img_size = img.cols();
        UByteIndexer indexer = img.createIndexer();
        for(int i=0;i<len_size;i++){
            for(int j=0;j<img_size;j++){
                histogram[indexer.get(i,j)]++;
            }
        }
        for(int i=1;i<256;i++)
            histogram[i] += histogram[i - 1];
        int i=0;
        while(i < h_size && histogram[i] <= flat_min)
            i += 1;
        p_min_pixel = i;

        i = h_size - 1;
        while (i > 0 && histogram[i] > flat_max)
            i -= 1;

        if (i < h_size - 1)
            i += 1;
        p_max_pixel = i;
        return new int[]{p_min_pixel, p_max_pixel};
    }

    public static int[] gray_minmax(Mat img){
        UByteIndexer index = img.createIndexer();
        int min_pixel = index.get(0,0),max_pixel = index.get(0,0);
        for(int i=0;i<img.rows();i++){
            for(int j=0;j<img.cols();j++){
                int value = index.get(i,j);
                if(value<min_pixel)
                    min_pixel = value;
                if(value>max_pixel)
                    max_pixel = value;
            }
        }
        return new int[]{min_pixel, max_pixel};
    }

    public static void gray_balance(Mat img, int flat_min, int flat_max)
    {
        int p_min_pixel=0;
        int p_max_pixel=0;

        if(img ==null || img.rows()==0)
            return;
        int img_size = img.rows() * img.cols();
        if(flat_min + flat_max > 2*img_size){
        flat_min = (img_size - 1) / 2;
        flat_max = (img_size - 1) / 2 ;
        System.out.println("需要被碾平的像素数目太大");
        System.out.println("使用(size - 1) / 2");
        }

        int size[];
        if(0 != flat_min || 0 != flat_max)
            size = gray_quantities(img, flat_min, flat_max);
        else
            size = gray_minmax(img);

        gray_rescale(img, size[0], size[1]);
    }

    public static void gray_rescale(Mat img, int flat_min, int flat_max){
        UByteIndexer indexer = img.createIndexer();
        int norm[]=new int[256];
        if(flat_max < flat_min) {
            for (int i = 0; i < img.rows(); i++)
            {
                for (int j = 0; j < img.cols(); j++)
                {
                    indexer.put(i,j,255/2);
                }
            }
        }
        int i = flat_min;
        while(i<flat_max){
            norm[i] = (int)((i - flat_min) * 255/ (flat_max - flat_min) + 0.5);
            i+=1;
        }

        i = flat_max;
        while(i<256)
        {
            norm[i] = 255;
            i += 1;
        }
        for(i=0;i<img.rows();i++){
            for(int j=0;j<img.cols();j++){
                int color = indexer.get(i,j);
                indexer.put(i,j,norm[color]);
            }
        }
    }

    public static Mat color_balance(Mat img, int smin, int smax){
        int img_size = img.rows()*img.cols();
        MatVector vector=new MatVector();
        split(img,vector);
        gray_balance(vector.get(0), (int)(img_size * (smin / 100.0)), (int)(img_size * (smax / 100.0)));
        gray_balance(vector.get(1), (int)(img_size * (smin / 100.0)), (int)(img_size * (smax / 100.0)));
        gray_balance(vector.get(2), (int)(img_size * (smin / 100.0)), (int)(img_size * (smax / 100.0)));
        Mat merged = img.clone();
        merge(vector,merged);
        return merged;
    }

    public static Mat beauty_face(Mat image){
        return beauty_face(image,0.8);
    }

    public static Mat beauty_face(Mat image, double p){

        opencv_objdetect.CascadeClassifier face_patterns = new opencv_objdetect.CascadeClassifier("haarcascade_frontalface_alt.xml");

        //Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//将图片缩小，加快检测速度

        Mat gray = image.clone();
        cvtColor(image,gray,COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        RectVector faces = new RectVector();
        face_patterns.detectMultiScale(gray,faces,1.1,2,0,new Size(30,30),new Size());
        Mat dst = image.clone();
        boolean is_beautied = false;
        for(int i=0;i<faces.size();i++)
        {
            Rect rect = faces.get(i);
            is_beautied = true;
            int x = rect.x(), y = rect.y(), w = rect.width(), h = rect.height();
            w *= 1.414;
            h *= 1.414;
            Mat roi_img = dst.adjustROI(x, y, x + w, y + h);
            int value1 = 3;
            int value2 = 1;
            int dx = value1 * 5;
            int fc = (int) (value1 * 12.5);
            Mat temp1 = roi_img.clone();
            bilateralFilter(roi_img, temp1, dx, fc, fc);
            Mat temp2 = subtract(temp1, roi_img).asMat();
            Mat temp3 = temp2.clone();
            GaussianBlur(temp2, temp3, new Size(2 * value2 - 1, 2 * value2 - 1), -1);
            Mat temp4 = add(roi_img,temp3).asMat();
            add(multiply(roi_img,1-p),multiply(temp4,p)).asMat().copyTo(roi_img);
        }
        if(!is_beautied){
            int value1 = 3;
            int value2 = 1;
            int dx = value1 * 5;
            int fc = (int)(value1 * 12.5);
            Mat temp1 = image.clone();
            bilateralFilter(image, temp1, dx, fc, fc);
            Mat temp2 = subtract(temp1, image).asMat();
            Mat temp3 = temp2.clone();
            GaussianBlur(temp2, temp3, new Size(2 * value2 - 1, 2 * value2 - 1), -1);
            Mat temp4 = add(image, temp3).asMat();
            dst = add(multiply(image,1-p),multiply(temp4,p)).asMat();
            // For the haven's sake, it is cpp, right?
        }

        return dst;
    }
}
