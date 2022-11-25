package dem.vaccae.opencvminist4android


import android.content.Context
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class OpenCVJNI {
    companion object {
        //加载动态库
        init {
            System.loadLibrary("opencvminist4android")
        }
    }

    fun initOnnxModel(context: Context, rawid: Int): Boolean {
        try {
            val onnxDir: File = File(context.filesDir, "onnx")
            if (!onnxDir.exists()) {
                onnxDir.mkdirs()
            }
            //判断模型是否存在是否存在，不存在复制过来
            val onnxfile: File = File(onnxDir, "dnnNet.onnx")
            if (onnxfile.exists()){
                return initOpenCVDNN(onnxfile.absolutePath)
            }else {
                // load cascade file from application resources
                val inputStream = context.resources.openRawResource(rawid)

                val os: FileOutputStream = FileOutputStream(onnxfile)
                val buffer = ByteArray(4096)
                var bytesRead: Int
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    os.write(buffer, 0, bytesRead)
                }
                inputStream.close()
                os.close()
                return initOpenCVDNN(onnxfile.absolutePath)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    //region JNI函数
    //初始化Dnn
    external fun initOpenCVDNN(onnxfilepath: String): Boolean

    //手写数字识别
    external fun ministDetector(bmp: Bitmap): MinistResult?

    //测试二值化图
    external fun thresholdBitmap(bmp: Bitmap): Bitmap
    //endregion


}