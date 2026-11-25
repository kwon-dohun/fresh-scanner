package com.freshscanner.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** 개체 사진 위에 판단에 쓴 부분(하양)과 결함 의심 부분(빨강) 마스크를 겹쳐 보여준다. */
class EvidenceView @JvmOverloads constructor(c: Context, a: AttributeSet? = null) : View(c, a) {
    private var photo: Bitmap? = null
    private var mask: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    fun set(photo: Bitmap, mask: Bitmap) {
        this.photo = photo
        this.mask = mask
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val p = photo ?: return
        val m = mask ?: return
        val side = min(width, height).toFloat()
        val l = (width - side) / 2f
        val dst = RectF(l, 0f, l + side, side)
        canvas.drawBitmap(p, null, dst, paint)
        canvas.drawBitmap(m, null, dst, paint)
    }
}
