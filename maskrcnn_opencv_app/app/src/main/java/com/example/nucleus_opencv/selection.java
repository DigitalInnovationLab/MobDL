package com.example.nucleus_opencv;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Calendar;

public class selection extends AppCompatActivity {

    private RadioGroup radioTypeGroup;
    private RadioButton radioTypeButton;
    private Button btnDisplay;
    private String ImagePath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_selection);
        ImagePath = getIntent().getStringExtra("IMAGE_PATH");
        addListenerOnButton();
    }

    public void addListenerOnButton() {

        radioTypeGroup = (RadioGroup) findViewById(R.id.radio);
        btnDisplay = (Button) findViewById(R.id.save);

        btnDisplay.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {

                // get selected radio button from radioGroup
                int selectedId = radioTypeGroup.getCheckedRadioButtonId();

                // find the radiobutton by returned id
                radioTypeButton = (RadioButton) findViewById(selectedId);

                Toast.makeText(selection.this,
                        radioTypeButton.getText()+"Saved", Toast.LENGTH_SHORT).show();

                String baseDir = android.os.Environment.getExternalStorageDirectory().toString();
                String fileName = "ParasiteData.csv";
                String filePath = baseDir + File.separator + fileName;

                File f = new File(filePath);
                CSVWriter writer;
                FileWriter mFileWriter;

                // File exist
                try {
                    if (f.exists() && !f.isDirectory()) {
                        mFileWriter = new FileWriter(filePath, true);
                        writer = new CSVWriter(mFileWriter);
                    } else {
                        writer = new CSVWriter(new FileWriter(filePath));
                    }
                    Log.i("Path", filePath);
                    String[] data = {ImagePath, radioTypeButton.getText().toString(), Calendar.getInstance().getTime().toString()};

                    writer.writeNext(data);

                    writer.close();
                }
                catch(IOException e) {
                    e.printStackTrace();
                }

            }

        });
    }





}
