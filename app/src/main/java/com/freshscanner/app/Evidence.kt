package com.freshscanner.app

import android.graphics.Bitmap
import android.graphics.Canvas
import com.freshscanner.app.core.Badge
import com.freshscanner.app.core.Frame
import com.freshscanner.app.core.FruitDetector

/** 탭한 개체의 확대 사진과 같은 영역의 마스크를 만든다. 둘 다 개체 중심에서 반지름의 1.5배씩 잘라 서로 겹친다. */
object Evidence {
    fun build(
        screen: Bitmap,
        frame: Frame,
        b: Badge,
        all: List<Badge>,
        cx: Float, cy: Float, rView: Float, // 화면 픽셀 기준 개체 위치와 반지름
    ): Pair<Bitmap, Bitmap> {
        val side = (3f * rView).toInt().coerceIn(24, 1200)
        val photo = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        Canvas(photo).drawBitmap(screen, -(cx - side / 2f), -(cy - side / 2f), null)

        val fw = frame.w; val fh = frame.h
        val fx = b.x * fw; val fy = b.y * fh; val r = b.r * fw
        val half = r * 1.5f
        val ms = (2 * half).toInt().coerceAtLeast(16)
        val x0 = (fx - half).toInt(); val y0 = (fy - half).toInt()
        val px = IntArray(ms * ms)
        for (j in 0 until ms) for (i in 0 until ms) {
            val x = x0 + i; val y = y0 + j
            if (x !in 0 until fw || y !in 0 until fh) continue
            val dx = x - fx; val dy = y - fy
            val d2 = dx * dx + dy * dy
            if (d2 > (1.15f * r) * (1.15f * r)) continue
            // 더 가까운 다른 개체의 픽셀이면 건너뛴다.
            if (all.any { o ->
                    o.id != b.id && (x - o.x * fw).let { ox -> ox * ox } + (y - o.y * fh).let { oy -> oy * oy } < d2
                }) continue
            px[j * ms + i] = when (FruitDetector.pixelClass(frame.argb[y * fw + x], d2 <= (0.9f * r) * (0.9f * r))) {
                1 -> 0x40FFFFFF
                2 -> 0xCCFF3B30.toInt()
                else -> 0
            }
        }
        return photo to Bitmap.createBitmap(px, ms, ms, Bitmap.Config.ARGB_8888)
    }
}
