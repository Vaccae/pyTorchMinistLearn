package dem.vaccae.opencvminist4android

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import dem.vaccae.opencvminist4android.databinding.ActivityMainBinding
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var isInitDNN: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        //初始化DNN
        isInitDNN = try {
            val jni = OpenCVJNI()
            val res = jni.initOnnxModel(this, R.raw.resnet)
            binding.tvshow.text = if(res){
                "OpenCV DNN初始化成功"
            }else{
                "OpenCV DNN初始化失败"
            }
            res
        } catch (e: Exception) {
            binding.tvshow.text = e.message
            false
        }

        binding.signatureView.setBackgroundColor(Color.rgb(245, 245, 245))

        binding.btnclear.setOnClickListener {
            binding.signatureView.clear()
        }

        binding.btnSave.setOnClickListener {
            if(!isInitDNN) return@setOnClickListener
            val bmp = binding.signatureView.getBitmapFromView()
            //处理图像
            val ministres:MinistResult? = try{
                val jni = OpenCVJNI()
                jni.ministDetector(bmp)
            }catch (e:Exception){
                binding.tvshow.text = e.message
                null
            }

            ministres?.let {
                binding.tvshow.text = it.msg
                binding.imgv.scaleType = ImageView.ScaleType.FIT_XY
                binding.imgv.setImageBitmap(it.bmp)
            }
        }


    }

}