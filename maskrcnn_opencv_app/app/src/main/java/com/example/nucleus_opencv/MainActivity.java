package com.example.nucleus_opencv;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    TextView textView;
    private ImageView image_view;
    private static final int PICK_IMAGE = 1;
    private int STORAGE_PERMISSON = 1001;
    private String IMAGE_PATH = "";
    private Net net;

    public String get_image_path() {
        return IMAGE_PATH;
    }
    public void set_image_path(String image_path) {
        this.IMAGE_PATH = image_path;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        long startTime = System.currentTimeMillis();
        String proto = getPath("maskrcnn.pbtxt", getBaseContext());
        String weights = getPath("nucleus_opencv.pb", getBaseContext());
        net = Dnn.readNetFromTensorflow(weights,proto);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        long endTime = System.currentTimeMillis();
        System.out.println("model timing " + (endTime-startTime)/1000);
        Log.i("OpenCV", "Network loaded successfully");

        // Example of a call to a native method
        textView = findViewById(R.id.maintext);
        textView.setText("model timing " + (endTime-startTime)/100);

        image_view = findViewById(R.id.imageView);
        image_view.setImageBitmap(getBitmapFromAsset(getBaseContext(),"Logo-Swinburne.jpg"));

        Button btn_choose_photo = findViewById(R.id.imageButton);
        btn_choose_photo.setOnClickListener(btnChoosePhotoPressed);//select photo

        Button btn_pred = findViewById(R.id.pred); // do prediction
        btn_pred.setOnClickListener(btnPredPressed);


        Button btn_sel = findViewById(R.id.selection); // do prediction
        btn_sel.setOnClickListener(btnSelPressed);

        findViewById(R.id.progressBar).setVisibility(View.INVISIBLE);//remove loading
        findViewById(R.id.selection).setVisibility(View.INVISIBLE);//remove loading
    }


    public View.OnClickListener btnChoosePhotoPressed = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            textView.setText("Parasite Application");
            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
                    String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                    requestPermissions(permissions, STORAGE_PERMISSON);

                } else {

                    Intent i = new Intent(Intent.ACTION_PICK,
                            android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                    final int ACTIVITY_SELECT_IMAGE = 1234;
                    startActivityForResult(i, ACTIVITY_SELECT_IMAGE);

                }
            }
            else{

                Intent i = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                final int ACTIVITY_SELECT_IMAGE = 1234;
                startActivityForResult(i, ACTIVITY_SELECT_IMAGE);
            }

        }
    };

    public View.OnClickListener btnPredPressed = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            long startTime = System.currentTimeMillis();
            Toast.makeText(MainActivity.this,
                    "Predicting...", Toast.LENGTH_SHORT).show();

            findViewById(R.id.progressBar).setVisibility(View.VISIBLE);//add loading

            String imagePath = get_image_path();
//            ProgressBar simpleProgressBar = findViewById(R.id.progressBar);
            Bitmap bmp = BitmapFactory.decodeFile(imagePath);
            Mat frame = new Mat();
            Bitmap bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(bmp32, frame);

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
            Imgproc.resize(frame, frame, new Size(512,512));

            Mat blob = Dnn.blobFromImage(frame,0 , new Size(frame.cols(),frame.rows()), new Scalar(127.5, 127.5, 127.5), true, false);
            net.setInput(blob);


            List<String> outNames = new ArrayList<String>();

            outNames.add("detection_out_final");
            outNames.add("detection_masks");

            List<Mat> outs = new ArrayList<Mat>();

            net.forward(outs,outNames);


            Mat frame2 = postprocess(frame, outs);

            Bitmap myBitmap;

            myBitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame2, myBitmap);

            long endTime = System.currentTimeMillis();
            System.out.println("Pred timing " + (endTime-startTime)/1000);
            textView = findViewById(R.id.maintext);
            textView.setText("Pred timing " + (endTime-startTime)/1000);

            findViewById(R.id.progressBar).setVisibility(View.INVISIBLE);//remove loading

            image_view = findViewById(R.id.imageView);
            image_view.setImageBitmap(myBitmap);
            findViewById(R.id.selection).setVisibility(View.VISIBLE);//remove loading



        }
    };


    public View.OnClickListener btnSelPressed = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            String imagePath = get_image_path();

            Intent i = new Intent(MainActivity.this,  selection.class);
            i.putExtra("IMAGE_PATH", imagePath);
            startActivity(i);
        }
    };

    public Mat postprocess(Mat frame, List<Mat> outs){


        Mat outDetections = outs.get(0);
        Mat outMasks = outs.get(1);

        int cols = frame.cols();
        int rows = frame.rows();
        // Output size of masks is NxCxHxW where
        // N - number of detected boxes
        // C - number of classes (excluding background)
        // HxW - segmentation shape
        int numDetections = outDetections.size(2);
        int numClasses = outMasks.size(1);
        long dump = outDetections.total() / 7;
        double confThreshold = 0.5; // Confidence threshold
        double maskThreshold = 0.3; // Mask threshold

        outDetections = outDetections.reshape(1, (int) dump);
        for (int i = 0; i < numDetections; ++i)
        {
            double confidence = outDetections.get(i, 2)[0];
            if (confidence > confThreshold)
            {
                // Extract the bounding box
                int classId = (int)outDetections.get(i, 1)[0];
                int left   = (int)(outDetections.get(i, 3)[0] * cols);
                int top    = (int)(outDetections.get(i, 4)[0] * rows);
                int right  = (int)(outDetections.get(i, 5)[0] * cols);
                int bottom = (int)(outDetections.get(i, 6)[0] * rows);


                // Draw rectangle around detected object.
                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                        new Scalar(0,0,255),2);
                String label = classNames[classId] + ": " + String.format("%.2f",(confidence*100));
                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 1, 2, baseLine);
                // Draw background for label.
//                Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
//                        new Point(left + labelSize.width, top + baseLine[0]),
//                        new Scalar(255,255,255));
                // Write class name and confidence.
                Imgproc.putText(frame, classNames[classId], new Point(left, top - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255),2);
                Log.i("Prediction",label );

                textView.setText(String.format("Parasite Application \n\n %s %%", label));

                // Extract the mask for the object
//                Mat objectMask(outMasks.size(2), outMasks.size(3), CV_32F, outMasks.ptr<float>(i,classId));

                // Draw bounding box, colorize and show the mask on the image
            }
        }

//        Size sv = new Size(1024,1024);
//        Imgproc.resize( frame, frame, sv );
        return frame;
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        switch(requestCode) {
            case 1234: // load the image
                if(resultCode == RESULT_OK){
                    Uri selectedImage = data.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};

                    Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                    cursor.moveToFirst();

                    int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                    String filePath = cursor.getString(columnIndex);
                    cursor.close();

                    image_view = findViewById(R.id.imageView);
                    image_view.setImageBitmap(BitmapFactory.decodeFile(filePath));

                    set_image_path(filePath);

                }
        }

    };

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == STORAGE_PERMISSON)  {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                Intent i = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                final int ACTIVITY_SELECT_IMAGE = 1234;
                startActivityForResult(i, ACTIVITY_SELECT_IMAGE);

            } else {
                Toast.makeText(this, "Permission DENIED", Toast.LENGTH_SHORT).show();
            }
        }
    }


    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i("OpenCv", "Failed to upload a file");
        }
        return "";
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }

        return bitmap;
    }


    // Used to load the 'native-lib' library on application startup.
    static {
//        System.loadLibrary("c++_shared");
        System.loadLibrary("native-lib");
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

//    public native String stringFromJNI();
//    public native void doPrediction(String imagePath);
    private static final String[] classNames = {"Nucleus"};
}
