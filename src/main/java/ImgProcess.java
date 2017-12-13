import org.bytedeco.javacpp.opencv_core.Mat;
import utils.ImageUtils;

import java.time.Clock;
import java.util.Map;

import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

/**
 * Created by Paindar on 17/12/12.
 */
public class ImgProcess
{
    public static void main(String[] args)
    {
        long str = System.currentTimeMillis();
        Map<Long,int[]> color_table = ImageUtils.init_color_table("./lookup-table.png");
        // It may be well if serialize color table.
        System.out.println("Initialization time= "+ (System.currentTimeMillis() - str));
        Mat image = imread("./PsiPollution3.jpg");
        imshow("origin image", image);
        if(image ==null || image.rows()==0){
            System.out.println("cannot open image.");
            return;
        }

        Mat flt = imread("lookup-table-yellow.png");
        imshow("with filter", flt);
        if(flt ==null || flt.rows()==0)
        {
            System.out.println("cannot open yellow color table.");
            return;
        }

        //Color balance
        Mat balance = ImageUtils.color_balance(image, 10, 95);
        imshow("with balance", balance);
        //You can resize it by @function cv2.resize
        //image = cv2.resize(image, (1280, 768))

        Mat dst = ImageUtils.beauty_face(image, 1);

        Mat with_filter = ImageUtils.add_filter(image, color_table, flt);
        imshow("with filter", with_filter);

        Mat overlay = imread("./overlay.png");
        if (overlay == null || overlay.rows()==0)
            System.out.println("cannot open overlay.");
        else
        {
            Mat with_overlay = ImageUtils.add_overlay(image, overlay, 0.3, 0.7);
            imshow("with overlay", with_overlay);
            Mat with_blur = ImageUtils.blur(image, 0.0057);
            imshow("with blur", with_blur);
        }
        imshow("hmmmm", dst);
        imwrite("output.jpg", dst);
        waitKey(0);
    }
}
