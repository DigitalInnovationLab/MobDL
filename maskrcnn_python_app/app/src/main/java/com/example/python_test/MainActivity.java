package com.example.python_test;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

public class MainActivity extends AppCompatActivity {


    TextView textView;
    private ImageView image_view;
    private static final int PICK_IMAGE = 1;
    private int STORAGE_PERMISSON = 1001;
    private String IMAGE_PATH = "";
    private PyObject pred_model;

    public String get_image_path() {
        return IMAGE_PATH;
    }
    public void set_image_path(String image_path) {
        this.IMAGE_PATH = image_path;
    }

    public PyObject get_pred_model() {
        return pred_model;
    }
    public void set_pred_model(PyObject Model) {
        this.pred_model = Model;
    }




    public View.OnClickListener btnChoosePhotoPressed = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
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




    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        switch(requestCode) {
            case 1234:
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

                    /* Now you have choosen image in Bitmap format in object "yourSelectedImage". You can use it in way you want! */
                }
        }

    };

    private class DownloadFilesTask extends AsyncTask<String, String, String> {
        protected String doInBackground(String... image_path) {
            Python py = Python.getInstance();
            publishProgress("Instanced loaded");

            PyObject pyf = py.getModule("prediction/parasite/test");

            PyObject image_pred2, pred_model;

            pred_model = get_pred_model();

            System.out.println(image_path[0]);
            long startTime = System.currentTimeMillis();
            image_pred2 = pyf.callAttr("test", image_path[0],pred_model);
            long endTime = System.currentTimeMillis();
            System.out.println("Pred timing " + (endTime-startTime)/1000);

            return String.valueOf((endTime-startTime)/1000);
        }

        @Override
        protected void onProgressUpdate(String... values) {
            super.onProgressUpdate(values);
            System.out.println(values[0]);
        }

        protected void onPostExecute(String result) {
            System.out.println("getting result in main thread: " + result);
            textView = findViewById(R.id.maintext);
            textView.setText(result);

        }
    }

    public View.OnClickListener btnPredPressed = new View.OnClickListener() {
        @Override
        public void onClick(View v) {

            String image_path = get_image_path();

            textView = findViewById(R.id.maintext);
            textView.setText(image_path.toString());


            new DownloadFilesTask().execute(image_path);


        }
    };




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button btn_choose_photo = findViewById(R.id.imageButton);
        btn_choose_photo.setOnClickListener(btnChoosePhotoPressed);

        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));

            Python py = Python.getInstance();
            PyObject pyf = py.getModule("prediction/parasite/test");

            PyObject pred_model;

            long startTime = System.currentTimeMillis();
            pred_model = pyf.callAttr("load_model");
            long endTime = System.currentTimeMillis();
            System.out.println("Model Loading time " + (endTime-startTime)/1000);

            System.out.println("Loaded model");
            set_pred_model(pred_model);//pass the model

            textView = findViewById(R.id.maintext);
            textView.setText("Model Loading time " + (endTime-startTime)/1000);


        }


        Button btn_pred = findViewById(R.id.pred); // Replace with id of your button.
        btn_pred.setOnClickListener(btnPredPressed);

        findViewById(R.id.loadingPanel).setVisibility(View.VISIBLE);//remove loading

    }


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
}
