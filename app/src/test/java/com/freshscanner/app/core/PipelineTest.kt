package com.freshscanner.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt
import kotlin.random.Random

/** 합성 이미지로 검출과 점수 파이프라인의 기본 동작을 확인한다. 실제 정확도는 실측으로 따로 잰다. */
class PipelineTest {
    private val w = 256
    private val h = 192
    private val bg = rgb(70, 78, 72)

    private fun rgb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    /** 배경 위에 주황 원을 그린다. spots가 true면 어두운 반점을 넣는다. */
    private fun scene(circles: List<Triple<Int, Int, Int>>, spots: Set<Int> = emptySet(), noise: Int = 0): Frame {
        val px = IntArray(w * h) { bg }
        val rnd = Random(7)
        circles.forEachIndexed { idx, (cx, cy, r) ->
            for (y in 0 until h) for (x in 0 until w) {
                val d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if (d2 <= r * r) {
                    val j = if (noise > 0) rnd.nextInt(-noise, noise + 1) else 0
                    px[y * w + x] = rgb(240 + j.coerceIn(-15, 15), 125 + j, 20)
                }
            }
            if (idx in spots) {
                // 반점: 원 안쪽에 어두운 덩어리 여러 개
                for ((dx, dy) in listOf(-6 to -4, 5 to 3, 0 to 8, -4 to 6)) {
                    for (y in -3..3) for (x in -3..3) {
                        if (x * x + y * y <= 9) px[(cy + dy + y) * w + (cx + dx + x)] = rgb(40, 30, 20)
                    }
                }
            }
        }
        return Frame(w, h, px)
    }

    @Test
    fun detectsSeparateFruitsAndIgnoresBackground() {
        val dets = FruitDetector.detect(scene(listOf(Triple(60, 60, 20), Triple(130, 90, 20), Triple(200, 130, 20))))
        assertEquals(3, dets.size)
    }

    @Test
    fun emptySceneGivesNoDetections() {
        assertTrue(FruitDetector.detect(scene(emptyList())).isEmpty())
    }

    @Test
    fun defectiveFruitGetsVetoAndRanksLast() {
        val frame = scene(listOf(Triple(60, 60, 22), Triple(130, 90, 22), Triple(200, 130, 22)), spots = setOf(1))
        val badges = Tracker().update(FruitDetector.detect(frame))
        assertEquals(3, badges.size)
        assertEquals("주의", badges.last().label)
        assertEquals(Tone.BEST, badges.first().tone)
    }

    @Test
    fun smoothUniformFruitOutscoresNoisyOne() {
        val clean = FruitDetector.detect(scene(listOf(Triple(80, 90, 24))))
        val noisy = FruitDetector.detect(scene(listOf(Triple(80, 90, 24)), noise = 40))
        assertEquals(1, clean.size)
        assertEquals(1, noisy.size)
        assertTrue(Scorer.score(clean[0].features).score > Scorer.score(noisy[0].features).score)
    }

    @Test
    fun trackerKeepsIdForSameFruitAcrossFrames() {
        val t = Tracker()
        val a = t.update(FruitDetector.detect(scene(listOf(Triple(80, 90, 22)))))
        val b = t.update(FruitDetector.detect(scene(listOf(Triple(82, 91, 22)))))
        assertEquals(a[0].id, b[0].id)
    }

    @Test
    fun hsvOfOrangeIsInRange() {
        val hh = FloatArray(1); val ss = FloatArray(1); val vv = FloatArray(1)
        rgbToHsv(rgb(240, 125, 20), hh, ss, vv, 0)
        assertTrue(hh[0] in 20f..35f)
        assertTrue(ss[0] > 0.9f)
        assertTrue(sqrt(vv[0]) > 0.9f)
    }
}
