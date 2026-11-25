package com.freshscanner.app.core

import java.nio.ByteBuffer

/** 분석용으로 줄인 정방향(회전 보정 완료) 프레임. argb는 0xAARRGGBB. */
class Frame(val w: Int, val h: Int, val argb: IntArray)

/**
 * 카메라 RGBA 버퍼를 outW 폭의 정방향 프레임으로 줄인다.
 * rotation은 ImageProxy.imageInfo.rotationDegrees (0/90/180/270).
 */
fun downsampleUpright(
    buf: ByteBuffer,
    rowStride: Int,
    pixelStride: Int,
    srcW: Int,
    srcH: Int,
    rotation: Int,
    outW: Int,
): Frame {
    val upW = if (rotation % 180 == 0) srcW else srcH
    val upH = if (rotation % 180 == 0) srcH else srcW
    val outH = (outW.toLong() * upH / upW).toInt()
    val out = IntArray(outW * outH)
    for (y in 0 until outH) {
        val uy = y * upH / outH
        for (x in 0 until outW) {
            val ux = x * upW / outW
            // 정방향 좌표 (ux, uy)를 원본 센서 좌표로 되돌린다. (프레임마다 수만 번 도는 곳이라 Pair를 만들지 않는다)
            val sx: Int
            val sy: Int
            when (rotation) {
                90 -> { sx = uy; sy = srcH - 1 - ux }
                180 -> { sx = srcW - 1 - ux; sy = srcH - 1 - uy }
                270 -> { sx = srcW - 1 - uy; sy = ux }
                else -> { sx = ux; sy = uy }
            }
            val o = sy * rowStride + sx * pixelStride
            val r = buf.get(o).toInt() and 0xFF
            val g = buf.get(o + 1).toInt() and 0xFF
            val b = buf.get(o + 2).toInt() and 0xFF
            out[y * outW + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    return Frame(outW, outH, out)
}
