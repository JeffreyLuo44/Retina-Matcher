import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.CLAHE;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

/*
 * COMPX301-22A Assignment 4
 * By Jedd Lupoy (1536884) and Jeffrey Luo (1535901)
 */

public class RetinalMatch {
    public static void main(String[] args) {
        // Check that the user has specified two images to compare match.
        if (args.length < 1) {
            System.out.println("USAGE: Please specify the path of two images files.");
            return;
        }
        // Loading OpenCV native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Create two instances of image processes.
        ImageProcess process1 = new ImageProcess(args[0]);
        ImageProcess process2 = new ImageProcess(args[1]);
        // Start image process for the first and second image.
        processPipeline(process1, "image1.jpg");
        processPipeline(process2, "image2.jpg");
        // Test whether first and second image are matching.
        testMatch(process1, process2);
    }

    /*
     * Begins the image manipulation process pipeline to detect retinal veins).
     */
    public static void processPipeline(ImageProcess image, String fileName) {
        image.resize(image._image, 60);
        image.applyCrop(image._result);
        image.addOpening(image._result);
        image.addSharpening(image._result);
        image.addGrayscale(image._result);
        image.addGaussianBlur(image._result);
        image.addAutoContrast(image._result);
        image.addLaplacian(image._result);
        image.addMedianBlur(image._edges);
        image.addGaussianBlur(image._result);
        image.addThresholding(image._result);
        image.addClosing(image._result);
        image.writeImageToFile(fileName);
    }


    // https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
    // https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html#matchtemplate
    // https://riptutorial.com/opencv/example/22915/template-matching-with-java
    // https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
    // https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    /*
     * Returns whether the two images match.
     */
    public static void testMatch(ImageProcess imageOne, ImageProcess imageTwo) {
        // How much the image is divided by.
        int splits = 4;
        // Keeps track of how many matches were made.
        int matchedCounter = 0;
        // Matches required to be considered a true match.
        int matchesRequired = 3;
        // Threshold for when part of an image is considered a match.
        double threshold = 0.45;
        // Create new matrix instances.
        Mat template = new Mat();
        Mat result = new Mat();

        // Iterate through all rows.  
        for (int i = 0; i < splits; i++) {
            // Iterate through all columns.
            for (int j = 0; j < splits; j++) {
                // Define where to cut the image.
                int[][] translateArr = {{1, 0, -(imageOne._result.cols() / splits) * j}, {0, 1, -(imageOne._result.rows() / splits) * i}};
                Mat translation = new Mat(2, 3, CvType.CV_32F);

                // Iterate and apply cut points.
                for (int row = 0; row < 2; row++) {
                    for (int col = 0; col < 3; col++) {
                        translation.put(row, col, translateArr[row][col]);
                    }
                }
                // Align imageTwo with imageOne cut-outs.
                Imgproc.warpAffine(imageOne._result, template, translation, new Size(imageOne._result.cols() / splits, imageOne._result.rows() / splits));
                // Compute for match value.
                Imgproc.matchTemplate(imageTwo._result, template, result, Imgproc.TM_CCOEFF_NORMED);
                MinMaxLocResult mmr = Core.minMaxLoc(result);
                // Part of an image matches if match value is over threshold.
                if (mmr.maxVal > threshold)
                    matchedCounter++;
            }
        }
        // Certain number of matches required to be considered two retinal scans are a match.
        if (matchedCounter >= matchesRequired)
            System.out.println("1");
        else
            System.out.println("0");
    }
}

/*
 * Pipeline for edge detection.
 */
class ImageProcess {
    // Stores the file name.
    String _file;
    // Matrices.
    public Mat _image, _edges, _result, _kernel;

    /*
     * Constructor for ImageProcess.
     */
    public ImageProcess(String file) {
        // Save file name.
        _file = file;
        try {
            // Initialise the matrices.
            _image = Imgcodecs.imread(_file);
            _edges = new Mat(_image.rows(), _image.cols(), _image.type());
            _result = new Mat(_image.rows(), _image.cols(), _image.type());
            _kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size((2*2) + 1, (2*2) + 1));
        } catch (Exception ex) {
            System.err.println("USAGE: Enter a valid image file.");
        }
    }

    /*
     * Reduces the size of the image by a scale value.
     */
    public void resize(Mat image, int scale){
        double width = (image.size().width * ((double) scale/100));
        double height = (image.size().height * ((double) scale / 100));
        Imgproc.resize(image, _result, new Size(width, height));
    }

    // https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
    /*
     * Sharpens the image.
     */
    public void addSharpening(Mat image) {
        Core.addWeighted(image, 2, _result, -0.5, 0, _result);
        // Imgcodecs.imwrite("sharpening.jpg", _result);
    }

    // https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    /*
     * Adds median blur.
     */
    public void addMedianBlur(Mat image) {
        Imgproc.medianBlur(image, _result, 7);
        // Imgcodecs.imwrite("median_blur.jpg", _result);
    }

    // https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    /*
     * Adds gaussian blur.
     */
    public void addGaussianBlur(Mat image) {
        Imgproc.GaussianBlur(image, _result, new Size(7, 7), 7); // BEST VERSION (7, 7), 7 (as of June 9, 2022)
        // Imgcodecs.imwrite("gaussian_blur.jpg", _result);
    }

    // https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    /*
     * Sets the image to grayscale.
     */
    public void addGrayscale(Mat image) {
        Imgproc.cvtColor(image, _result, Imgproc.COLOR_BGR2GRAY);
        // Imgcodecs.imwrite("grayscale.jpg", _result);
    }

    // https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e
    /*
     * Makes the retinal veins more clear.
     */
    public void addAutoContrast(Mat image) {
        // Imgproc.equalizeHist(image, _result);
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.apply(image, _result);
        // Imgcodecs.imwrite("auto_contrast.jpg", _result);
    }

    // https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
    /*
     * Edge detection to get the retinal veins.
     */
    public void addLaplacian(Mat image) {
        Imgproc.Laplacian(image, _edges, CvType.CV_8UC1, 5, 1, 1, Core.BORDER_DEFAULT); // BEST VERSION 5, 1, 1 (as of June 9, 2022)
        // Imgcodecs.imwrite("laplacian.jpg", _edges);
    }

    /*
     * Crops the image to remove background and focus on the retinal scan.
     */
    public void applyCrop(Mat image) {
        int[][] translateArr = {{1, 0, -(image.cols() / 7)}, {0, 1, -(image.rows() / 13)}};
        Mat translation = new Mat(2, 3, CvType.CV_32F);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                translation.put(i, j, translateArr[i][j]);
            }
        }
        Imgproc.warpAffine(image, _result, translation, new Size(image.cols() - (image.cols() / 19 * 6), image.rows() - image.rows() / 7));
    }

    // https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb <-- Erode
    // https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c <-- Dilate
    /*
     * Remove more noise.
     */
    public void addOpening(Mat image) {
        Imgproc.erode(image, _result, _kernel);
        Imgproc.dilate(_result, _result, _kernel);
        // Imgcodecs.imwrite("opening.jpg", _result);
    }

    /*
     * Connect retinal vein edges. 
     */
    public void addClosing(Mat image) {
        Imgproc.dilate(image, _result, _kernel);
        Imgproc.erode(_result, _result, _kernel);
        // Imgcodecs.imwrite("closing.jpg", _result);
    }

    // https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    /*
     * Remove image noise, and invert edges to black and background to white.
     */
    public void addThresholding(Mat image) {
        Imgproc.threshold(image, _result, 100, 255, Imgproc.THRESH_BINARY_INV); // BEST VERSION 40, 255 (as of June 9, 2022)
        // Imgcodecs.imwrite("thresholding.jpg", _result);
    }

    /*
     * Save processed image as a file.
     */
    public void writeImageToFile(String fileName) {
        Imgcodecs.imwrite(fileName, _result);
    }
}