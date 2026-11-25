package com.freshscanner.app.core

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer

/** 센서 프레임의 한 점이 회전 보정 뒤 정방향 프레임의 올바른 위치로 가는지 확인한다. */
class RotationTest {
    private val srcW = 8
    private val srcH = 4

    /** 센서 좌표 (6, 1)에만 빨간 픽셀이 있는 RGBA 버퍼. */
    private fun marker(): ByteBuffer {
        val buf = ByteBuffer.allocate(srcW * srcH * 4)
        val o = (1 * srcW + 6) * 4
        buf.put(o, 255.toByte()); buf.put(o + 3, 255.toByte())
        return buf
    }

    private fun redAt(rotation: Int): Pair<Int, Int> {
        val upW = if (rotation % 180 == 0) srcW else srcH
        val f = downsampleUpright(marker(), srcW * 4, 4, srcW, srcH, rotation, upW)
        val i = f.argb.indexOfFirst { (it shr 16) and 0xFF == 255 }
        return (i % f.w) to (i / f.w)
    }

    @Test fun rotation0KeepsPosition() = assertEquals(6 to 1, redAt(0))
    @Test fun rotation90() = assertEquals(2 to 6, redAt(90))
    @Test fun rotation180() = assertEquals(1 to 2, redAt(180))
    @Test fun rotation270() = assertEquals(1 to 1, redAt(270))
}
