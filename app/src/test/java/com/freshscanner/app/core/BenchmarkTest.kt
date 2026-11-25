package com.freshscanner.app.core

import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.hypot
import kotlin.random.Random

/**
 * 합성 장면으로 검출 성능(정밀도/재현율)과 처리 속도를 잰다. 결과는 표준 출력에 찍는다.
 * 합성 이미지라 실제 시장 정확도가 아니다. 개선 전후를 같은 잣대로 비교하는 용도다.
 */
class BenchmarkTest {
    private val w = 256
    private val h = 192

    private fun rgb(r: Int, g: Int, b: Int) =
        (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)

    /** truth는 (x, y, 반지름)이고 x는 폭, y는 높이, 반지름은 폭 기준 0..1. */
    private class Scene(val name: String, val frame: Frame, val truth: List<Triple<Float, Float, Float>>)

    private fun scene(
        name: String,
        centers: List<Pair<Int, Int>>,
        r: Int = 20,
        radii: List<Int>? = null,
        cast: Triple<Float, Float, Float> = Triple(1f, 1f, 1f),
        dim: Float = 1f,
        noise: Int = 3,
        clutter: Boolean = false,
    ): Scene {
        val rnd = Random(11)
        val px = IntArray(w * h)
        fun put(i: Int, rr: Float, g: Float, b: Float) {
            val j = if (noise > 0) rnd.nextInt(-noise, noise + 1) else 0
            px[i] = rgb(
                ((rr + j) * cast.first * dim).toInt(),
                ((g + j) * cast.second * dim).toInt(),
                ((b + j) * cast.third * dim).toInt(),
            )
        }
        for (i in 0 until w * h) put(i, 70f, 78f, 72f)
        if (clutter) {
            // 주황 포장재(사각형)와 작은 주황 점
            for (y in 140..175) for (x in 10..80) put(y * w + x, 235f, 120f, 25f)
            for (k in 0 until 6) put(rnd.nextInt(h) * w + rnd.nextInt(w), 235f, 120f, 25f)
        }
        centers.forEachIndexed { idx, (cx, cy) ->
            val rr = radii?.get(idx) ?: r
            for (y in 0 until h) for (x in 0 until w) {
                val d = hypot((x - cx).toFloat(), (y - cy).toFloat())
                if (d <= rr) {
                    val s = 1f - 0.3f * (d / rr) * (d / rr)
                    put(y * w + x, 240f * s, 128f * s, 24f * s)
                }
            }
        }
        val truth = centers.mapIndexed { i, c -> Triple(c.first.toFloat() / w, c.second.toFloat() / h, (radii?.get(i) ?: r).toFloat() / w) }
        return Scene(name, Frame(w, h, px), truth)
    }

    private val sep = listOf(40 to 40, 110 to 55, 180 to 40, 50 to 120, 120 to 130, 200 to 110)
    private val touching = listOf(40 to 50, 78 to 50, 130 to 100, 168 to 100, 60 to 150, 98 to 150)
    private val pile = listOf(60 to 60, 98 to 60, 136 to 60, 79 to 93, 117 to 93, 155 to 93, 60 to 126, 98 to 126, 136 to 126, 117 to 159)

    private fun scenes() = listOf(
        scene("깨끗함 6개", sep),
        scene("따뜻한 조명", sep, cast = Triple(1.2f, 1.0f, 0.7f)),
        scene("차가운 조명", sep, cast = Triple(0.8f, 1.0f, 1.25f)),
        scene("어두움", sep, dim = 0.55f),
        scene("잡음 많음", sep, noise = 18),
        scene("맞닿은 쌍 3개", touching),
        scene("빽빽한 더미 10개", pile),
        scene("주황 포장재 방해", sep.take(4), clutter = true),
        scene("크기 제각각 6개", sep, radii = listOf(12, 26, 17, 22, 14, 24)),
        scene("가장자리 잘림", listOf(8 to 40, 248 to 60, 120 to 4, 130 to 100, 60 to 186)),
    )

    @Test
    fun detectionQuality() {
        var tp = 0; var fp = 0; var fn = 0
        println("== 검출 벤치마크 (정답 대비) ==")
        for (s in scenes()) {
            val dets = FruitDetector.detect(s.frame)
            val matched = BooleanArray(dets.size)
            var hit = 0
            for (t in s.truth) {
                val k = dets.indices.firstOrNull { !matched[it] && hypot(dets[it].x - t.first, (dets[it].y - t.second) * h / w) < t.third } ?: -1
                if (k >= 0) { matched[k] = true; hit++ }
            }
            val miss = s.truth.size - hit
            val extra = matched.count { !it }
            tp += hit; fn += miss; fp += extra
            println("%-14s 정답 %2d  맞춤 %2d  놓침 %2d  헛검출 %2d".format(s.name, s.truth.size, hit, miss, extra))
        }
        val p = tp.toFloat() / (tp + fp).coerceAtLeast(1)
        val r = tp.toFloat() / (tp + fn).coerceAtLeast(1)
        println("전체 정밀도 %.2f  재현율 %.2f  F1 %.2f".format(p, r, 2 * p * r / (p + r).coerceAtLeast(1e-6f)))
    }

    @Test
    fun speed() {
        val frame = scene("속도용", sep).frame
        repeat(30) { FruitDetector.detect(frame) }
        val n = 100
        val t0 = System.nanoTime()
        repeat(n) { FruitDetector.detect(frame) }
        val detectMs = (System.nanoTime() - t0) / 1e6 / n

        // 1280x960 RGBA, 90도 회전 다운샘플 (실제 카메라 프레임 크기)
        val src = ByteBuffer.allocateDirect(1280 * 960 * 4)
        repeat(30) { downsampleUpright(src, 1280 * 4, 4, 1280, 960, 90, 256) }
        val t1 = System.nanoTime()
        repeat(n) { downsampleUpright(src, 1280 * 4, 4, 1280, 960, 90, 256) }
        val downMs = (System.nanoTime() - t1) / 1e6 / n
        println("== 속도 (JVM, 상대 비교용) ==")
        println("다운샘플 %.2f ms  검출 %.2f ms  합계 %.2f ms".format(downMs, detectMs, downMs + detectMs))
    }
}
