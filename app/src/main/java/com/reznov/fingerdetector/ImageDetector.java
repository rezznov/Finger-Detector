package com.reznov.fingerdetector;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ImageDetector extends AppCompatActivity {

    Button loadBtn;
    Button convertButton;
    ImageView imgPreview;
    ImageView imgConverted;


    Net tinyYolo;

    Bitmap bitmap;

    File photo;

    Uri photoUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detector);

        initView();
        photo = new File(Environment.getExternalStorageDirectory(), "Pic.jpg");

    }

    void initView() {
        loadBtn = findViewById(R.id.load);
        convertButton = findViewById(R.id.detect);
        imgPreview = findViewById(R.id.imgPreivew);
        imgConverted = findViewById(R.id.imgConverted);

        loadBtn.setOnClickListener(v -> {

            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//            photoUri = Uri.fromFile(photo);
            photoUri = FileProvider.getUriForFile(ImageDetector.this, BuildConfig.APPLICATION_ID + ".provider",photo);
            intent.putExtra(MediaStore.EXTRA_OUTPUT,photoUri);
            startActivityForResult(intent, 100);
        });
        convertButton.setOnClickListener(this::convertImage);
    }

    @Override
    protected void onResume() {
        super.onResume();
        initModel();
    }

    void initModel() {
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "مشکل در بارگزاری مدل هوش مصنوعی", Toast.LENGTH_SHORT).show();
        } else {
//            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov4-fingers.cfg";
            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov4-10000.cfg";
//            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/custom-yolov4-tiny-detector_best.weights";
            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov4-10000.weights";

            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
        }
    }

    void loadImage() {
//        bitmap = BitmapFactory.decodeFile(Environment.getExternalStorageDirectory() + "/test-model/three_2.jpg");
        bitmap = Bitmap.createScaledBitmap(rotateImage(BitmapFactory.decodeFile(photo.getAbsolutePath()), 90), 416, 416, true);
        imgPreview.setImageBitmap(bitmap);
    }

    void convertImage(View view) {
        Bitmap result = convert(bitmapToMat(bitmap));
        imgConverted.setImageBitmap(result);
    }

    Bitmap convert(Mat mat) {
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB);
        Mat imageBlob = Dnn.blobFromImage(mat, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
//            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
        tinyYolo.setInput(imageBlob);
        java.util.List<Mat> result = new java.util.ArrayList<Mat>(3);
        List<String> outBlobNames = new java.util.ArrayList<>();
//        outBlobNames.add(0, "yolo_139");
//        outBlobNames.add(1, "yolo_150");
//        outBlobNames.add(2, "yolo_161");
        outBlobNames.add(0, "yolo_30");
        outBlobNames.add(1, "yolo_37");
        tinyYolo.forward(result, outBlobNames);
        float confThreshold = 0.1f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        List<Rect2d> rect2ds = new ArrayList<>();
        Log.d("rrrr", "onCameraFrame: result size" + result.size());
        for (int i = 0; i < result.size(); ++i) {
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * mat.cols());
                    int centerY = (int) (row.get(0, 1)[0] * mat.rows());
                    int width = (int) (row.get(0, 2)[0] * mat.cols());
                    int height = (int) (row.get(0, 3)[0] * mat.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);
                    rects.add(new Rect(left, top, width, height));
                    rect2ds.add(new Rect2d(centerX, centerY, width, height));
                }
            }
        }
        int ArrayLength = confs.size();
        for (Float con : confs)
            Log.d("rrrr", "onCameraFrame: " + con);
        if (ArrayLength >= 1) {
            // Apply non-maximum suppression procedure.
            float nmsThresh = 0.2f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            Rect[] boxesArray = rects.toArray(new Rect[0]);
//            MatOfRect boxes = new MatOfRect(boxesArray);
            MatOfRect2d boxes = new MatOfRect2d(rect2ds.toArray(new Rect2d[0]));
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);
            // Draw result boxes:
            int[] ind = indices.toArray();
            for (int i = 0; i < ind.length; ++i) {
                int idx = ind[i];
                Rect box = boxesArray[idx];
                int idGuy = clsIds.get(idx);
                float conf = confs.get(idx);
//                    List<String> cocoNames = Arrays.asList("a person", "a bicycle", "a motorbike", "an airplane", "a bus", "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", "a stop sign", "a parking meter", "a car", "a bench", "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", "a kite", "a baseball bat", "a baseball glove", "a skateboard", "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", "a doughnut", "a cake", "a chair", "a sofa", "a potted plant", "a bed", "a dining table", "a toilet", "a TV monitor", "a laptop", "a computer mouse", "a remote control", "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", "a sink", "a refrigerator", "a book", "a clock", "a vase", "a pair of scissors", "a teddy bear", "a hair drier", "a toothbrush");
                List<String> cocoNames = Arrays.asList("one", "two", "three", "four", "five");
                int intConf = (int) (conf * 100);
                Imgproc.putText(mat, cocoNames.get(idGuy) + " " + intConf + "%", box.tl(), 0, 2, new Scalar(255, 255, 0), 2);
                Log.d("rrrr", "class neme : " + cocoNames.get(idGuy));
                Imgproc.rectangle(mat, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
            }
        }
        return matToBitmap(mat);
    }

    Bitmap matToBitmap(Mat mat) {
        Bitmap bmp = null;
        Mat tmp = new Mat(mat.rows(), mat.cols(), CvType.CV_8U, new Scalar(4));
        try {
            //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
            Imgproc.cvtColor(mat, tmp, Imgproc.COLOR_RGBA2RGB);
            bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(tmp, bmp);
            return bmp;
        } catch (CvException e) {
            Log.d("Exception", e.getMessage());
        }
        return null;
    }

    Mat bitmapToMat(Bitmap bitmap) {
        Mat mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if ((resultCode == Activity.RESULT_OK) && requestCode == 100)
            loadImage();
    }

    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }
}
