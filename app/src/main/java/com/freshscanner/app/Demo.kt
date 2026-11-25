package com.freshscanner.app

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.Shader
import kotlin.random.Random

/**
 * 디버그 빌드에서만 켜지는 데모 장면 (adb shell am start ... --ez demo true).
 * 카메라 대신 합성 감귤 사진을 같은 분석 파이프라인에 넣어, 에뮬레이터에서도 화면과 시트를 확인한다.
 */
object Demo {
    fun requested(a: Activity) =
        (a.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0 && a.intent.getBooleanExtra("demo", false)

    private class Fruit(val x: Float, val y: Float, val r: Float, val base: Int, val spots: Int = 0)

    fun bitmap(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.rgb(58, 70, 66))
        val fruits = listOf(
            Fruit(250f, 250f, 74f, Color.rgb(240, 122, 22)),
            Fruit(390f, 235f, 72f, Color.rgb(236, 128, 26)),
            Fruit(535f, 262f, 70f, Color.rgb(244, 118, 20)),
            Fruit(240f, 430f, 72f, Color.rgb(228, 150, 42)),
            Fruit(385f, 420f, 76f, Color.rgb(240, 124, 24)),
            Fruit(530f, 445f, 70f, Color.rgb(238, 126, 24), spots = 5),
            Fruit(255f, 610f, 73f, Color.rgb(242, 120, 22)),
            Fruit(395f, 600f, 72f, Color.rgb(234, 138, 34)),
            Fruit(540f, 620f, 70f, Color.rgb(240, 124, 24)),
            Fruit(245f, 790f, 72f, Color.rgb(240, 124, 24)),
            Fruit(390f, 800f, 74f, Color.rgb(226, 160, 52)),
            Fruit(535f, 795f, 70f, Color.rgb(242, 120, 22)),
        )
        val rnd = Random(5)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        for (f in fruits) {
            val light = scale(f.base, 1.12f)
            val dark = scale(f.base, 0.55f)
            p.shader = RadialGradient(f.x - f.r * 0.3f, f.y - f.r * 0.35f, f.r * 1.5f, intArrayOf(light, f.base, dark), floatArrayOf(0f, 0.6f, 1f), Shader.TileMode.CLAMP)
            c.drawCircle(f.x, f.y, f.r, p)
            p.shader = null
            p.color = Color.rgb(45, 32, 22)
            repeat(f.spots) {
                val a = rnd.nextFloat() * 6.283f
                val d = rnd.nextFloat() * f.r * 0.6f
                c.drawCircle(f.x + d * kotlin.math.cos(a), f.y + d * kotlin.math.sin(a), f.r * (0.07f + rnd.nextFloat() * 0.05f), p)
            }
        }
        return bmp
    }

    private fun scale(color: Int, k: Float) = Color.rgb(
        (Color.red(color) * k).toInt().coerceIn(0, 255),
        (Color.green(color) * k).toInt().coerceIn(0, 255),
        (Color.blue(color) * k).toInt().coerceIn(0, 255),
    )
}
