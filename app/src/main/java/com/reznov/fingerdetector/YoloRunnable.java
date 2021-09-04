package com.reznov.fingerdetector;

import android.os.Handler;

import org.opencv.core.Mat;

public class YoloRunnable implements Runnable {

    public Mat preMat;
    public Mat postMat;

    private Handler handler = new Handler();

    @Override
    public void run() {
//        handler.postDelayed(() -> ,50);
    }
}
