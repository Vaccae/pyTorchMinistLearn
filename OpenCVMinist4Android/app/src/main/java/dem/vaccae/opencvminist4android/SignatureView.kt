package dem.vaccae.opencvminist4android

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.core.view.drawToBitmap
import java.lang.Float.max
import kotlin.math.min

class SignatureView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    //笔划宽度
    private var STROKE_WIDTH = 5f

    private var lastTouchX = -1f
    private var lastTouchY = -1f

    //定义画笔相关
    private val paint = Paint()

    //定义画笔路径
    private val path = Path()

    //绘制的矩形区域
    private val drawRect = RectF()

    //初始化数据
    init {
        //设置抗锯齿
        paint.isAntiAlias = true
        //设置画笔颜色
        paint.color = Color.BLUE
        //设置画笔类型
        paint.style = Paint.Style.STROKE
        //设置画笔的线冒样式
        paint.strokeCap = Paint.Cap.ROUND
        //设置画笔连接处样式
        paint.strokeJoin = Paint.Join.ROUND
        //设置画笔宽度
        paint.strokeWidth = STROKE_WIDTH
    }

    //清除绘制
    fun clear(){
        path.reset()
        postInvalidate()
    }

    //获取当前页面图片
    fun getBitmapFromView():Bitmap{
        return drawToBitmap()
    }

    //设置笔划宽度
    fun setPaintStrokeWidth(width:Float){
        STROKE_WIDTH = width
        paint.strokeWidth = STROKE_WIDTH
    }

    //设置笔划宽度
    fun setPaintColor(color: Int){
        paint.color = color
    }

    //画笔手执处理
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        event?.let {
            val event_x = it.x
            val event_y = it.y

            when (it.action) {
                MotionEvent.ACTION_DOWN -> {
                    //点击按下时开始记录路径
                    path.moveTo(event_x, event_y)
                    //记录最后的X和Y的坐标
                    lastTouchX = event_x
                    lastTouchY = event_y

                    return true
                }
                MotionEvent.ACTION_MOVE, MotionEvent.ACTION_UP -> {
                    //计算绘制区域
                    drawRect.left = min(lastTouchX, event_x)
                    drawRect.right = max(lastTouchX, event_x)
                    drawRect.top = min(lastTouchY, event_y)
                    drawRect.bottom = max(lastTouchY, event_y)

                    // 当硬件跟踪事件的速度快于事件的交付速度时
                    // 事件将包含这些跳过点的历史记录
                    val historySize = it.historySize
                    (0 until historySize).forEach { i ->
                        val historicalx = it.getHistoricalX(i)
                        val historicaly = it.getHistoricalY(i)
                        if (historicalx < drawRect.left) {
                            drawRect.left = historicalx
                        } else if (historicalx > drawRect.right) {
                            drawRect.right = historicalx
                        }

                        if (historicaly < drawRect.top) {
                            drawRect.top = historicaly
                        } else if (historicaly > drawRect.bottom) {
                            drawRect.bottom = historicaly
                        }

                        path.lineTo(historicalx, historicaly)
                    }

                    // 回放历史记录后，将线路连接到触点。
                    path.lineTo(event_x, event_y)
                }
                else -> {
                    return false
                }
            }

            // 绘制时根据笔画宽度除2用于在中心开妈绘制
            postInvalidate(
                (drawRect.left - STROKE_WIDTH / 2).toInt(),
                (drawRect.top - STROKE_WIDTH / 2).toInt(),
                (drawRect.right + STROKE_WIDTH / 2).toInt(),
                (drawRect.bottom + STROKE_WIDTH / 2).toInt()
            );

            lastTouchX = event_x;
            lastTouchY = event_y;
        }
        return super.onTouchEvent(event)
    }


    override fun onDraw(canvas: Canvas?) {
        canvas?.let {
            it.drawPath(path, paint)
        }
        super.onDraw(canvas)
    }

}