package com.freshscanner.app.core

import org.junit.Assert.assertEquals
import org.junit.Test

/** 증거 마스크가 쓰는 픽셀 분류가 검출기의 판단 기준과 같은지 확인한다. */
class PixelClassTest {
    private fun rgb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    @Test fun orangeIsMeasured() = assertEquals(1, FruitDetector.pixelClass(rgb(240, 125, 20), true))
    @Test fun darkSpotInsideIsDefect() = assertEquals(2, FruitDetector.pixelClass(rgb(40, 30, 20), true))
    @Test fun darkSpotOutsideInnerIsIgnored() = assertEquals(0, FruitDetector.pixelClass(rgb(40, 30, 20), false))
    @Test fun grayBackgroundIsIgnored() = assertEquals(0, FruitDetector.pixelClass(rgb(70, 78, 72), true))
}
