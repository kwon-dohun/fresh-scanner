package com.freshscanner.app.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** 개체 하나에서 화면 픽셀로 계산한 원시 특징. 실측 기록(JSON)에 그대로 남겨 나중에 보정한다. */
data class Features(
    val meanH: Float,      // 색상 평균 (도)
    val stdH: Float,       // 색상 표준편차 (균일도)
    val meanS: Float,
    val meanV: Float,
    val greenFrac: Float,  // 연두빛 픽셀 비율 (덜 익음)
    val smooth: Float,     // 밝기 거칠기 (낮을수록 매끈)
    val axisRatio: Float,  // 타원 장축/단축 (편평비 대용)
    val defectFrac: Float, // 어두운 반점, 흰 곰팡이빛 비율
)

/** x, y는 프레임 폭·높이 기준 0..1, r은 프레임 폭 기준 반지름. */
data class Detection(val x: Float, val y: Float, val r: Float, val features: Features)

/**
 * v0: 색 기반 감귤 검출. 마스크의 거리 변환 봉우리를 개체 중심으로 삼아 맞닿은 과일을 나눈다.
 * 학습 데이터가 쌓이면 이 함수만 학습 모델로 바꾼다.
 * 한계: 주황 포장재·배경도 잡을 수 있고, 틈 없이 완전히 붙은 과일은 하나로 합쳐진다.
 * 분석 스레드 하나에서만 부른다 (버퍼를 재사용한다).
 */
object FruitDetector {
    private const val H_MIN = 8f
    private const val H_MAX = 55f
    private const val S_MIN = 0.45f
    private const val V_MIN = 0.35f
    private const val MIN_R = 6          // 개체로 칠 최소 반지름 (프레임 픽셀). 작은 갈색 그림자를 거른다
    private const val OWN_REACH = 1.35f  // 시드 반지름의 이 배수까지를 그 개체 몫으로 본다
    private const val SEPARATION = 1.15f // 시드끼리 최소 간격 (반지름 배수)
    private const val MAX_SEEDS = 64
    private const val DARK_V = 0.28f     // 이보다 어두우면 반점 의심
    private const val WHITE_S = 0.2f     // 채도가 이보다 낮고
    private const val WHITE_V = 0.6f     // 밝으면 곰팡이빛 의심

    // 프레임 크기가 같으면 매 프레임 같은 배열을 다시 쓴다.
    private var cap = 0
    private var hh = FloatArray(0)
    private var ss = FloatArray(0)
    private var vv = FloatArray(0)
    private var mask = BooleanArray(0) // 엄격한 색 기준: 색 통계와 결함 계산에 쓴다
    private var sil = BooleanArray(0)  // 느슨한 기준: 어두운 반점도 과일 안으로 보는 실루엣 (시드 분리용)
    private var dt = IntArray(0)

    private class Seeds {
        val x = FloatArray(MAX_SEEDS)
        val y = FloatArray(MAX_SEEDS)
        val r = FloatArray(MAX_SEEDS)
        var n = 0
    }

    @Synchronized
    fun detect(f: Frame): List<Detection> {
        val w = f.w
        val h = f.h
        val n = w * h
        if (n != cap) {
            cap = n
            hh = FloatArray(n); ss = FloatArray(n); vv = FloatArray(n)
            mask = BooleanArray(n); sil = BooleanArray(n); dt = IntArray(n)
        }
        for (i in 0 until n) {
            rgbToHsv(f.argb[i], hh, ss, vv, i)
            val inHue = hh[i] in H_MIN..H_MAX
            mask[i] = inHue && ss[i] >= S_MIN && vv[i] >= V_MIN
            sil[i] = inHue && ss[i] >= S_MIN && vv[i] >= 0.12f // 채도는 그대로 두고 어두운 반점만 안으로 들인다
        }
        distanceTransform(w, h)
        val seeds = pickSeeds(w, h)
        merge(seeds)

        val out = ArrayList<Detection>()
        for (k in 0 until seeds.n) {
            val feat = features(k, seeds, w, h) ?: continue
            out.add(Detection(seeds.x[k] / w, seeds.y[k] / h, seeds.r[k] / w, feat))
        }
        return out
    }

    /** 3-4 챔퍼 거리 변환. 단위는 픽셀의 3배. 프레임 밖은 배경으로 본다. */
    private fun distanceTransform(w: Int, h: Int) {
        val big = 1 shl 20
        for (i in 0 until w * h) dt[i] = if (sil[i]) big else 0
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (dt[i] == 0) continue
            var m = dt[i]
            m = min(m, if (x > 0) dt[i - 1] + 3 else 3)
            if (y > 0) {
                m = min(m, dt[i - w] + 3)
                m = min(m, if (x > 0) dt[i - w - 1] + 4 else 4)
                m = min(m, if (x < w - 1) dt[i - w + 1] + 4 else 4)
            } else m = min(m, 3)
            dt[i] = m
        }
        for (y in h - 1 downTo 0) for (x in w - 1 downTo 0) {
            val i = y * w + x
            if (dt[i] == 0) continue
            var m = dt[i]
            m = min(m, if (x < w - 1) dt[i + 1] + 3 else 3)
            if (y < h - 1) {
                m = min(m, dt[i + w] + 3)
                m = min(m, if (x < w - 1) dt[i + w + 1] + 4 else 4)
                m = min(m, if (x > 0) dt[i + w - 1] + 4 else 4)
            } else m = min(m, 3)
            dt[i] = m
        }
    }

    /** 거리 변환 값이 큰 순서로 훑으며, 이미 뽑은 시드와 가깝지 않은 봉우리를 시드로 삼는다. */
    private fun pickSeeds(w: Int, h: Int): Seeds {
        val minD = 3 * MIN_R
        var cnt = 0
        for (i in 0 until w * h) if (dt[i] >= minD) cnt++
        val keys = LongArray(cnt)
        var c = 0
        for (i in 0 until w * h) if (dt[i] >= minD) keys[c++] = (dt[i].toLong() shl 32) or i.toLong()
        keys.sort()

        val s = Seeds()
        for (j in cnt - 1 downTo 0) {
            if (s.n == MAX_SEEDS) break
            val i = (keys[j] and 0xFFFFFFFFL).toInt()
            val x = (i % w).toFloat()
            val y = (i / w).toFloat()
            val r = (keys[j] shr 32).toFloat() / 3f
            var near = false
            for (k in 0 until s.n) {
                val dx = x - s.x[k]; val dy = y - s.y[k]
                val lim = SEPARATION * s.r[k]
                if (dx * dx + dy * dy < lim * lim) { near = true; break }
            }
            if (!near) { s.x[s.n] = x; s.y[s.n] = y; s.r[s.n] = r; s.n++ }
        }
        return s
    }

    /**
     * 한 과일이 반점이나 프레임 가장자리 때문에 여러 시드로 쪼개진 것을 다시 합친다.
     * 맞닿은 서로 다른 과일은 중심 간격이 반지름 합과 거의 같고(약 1.0),
     * 쪼개진 조각은 훨씬 가깝다(0.6~0.8). 0.8 아래면 같은 개체로 본다.
     */
    private fun merge(s: Seeds) {
        val n = s.n
        val parent = IntArray(n) { it }
        fun find(a: Int): Int { var x = a; while (parent[x] != x) x = parent[x]; return x }
        for (i in 0 until n) for (j in i + 1 until n) {
            val dx = s.x[i] - s.x[j]; val dy = s.y[i] - s.y[j]
            val lim = 0.8f * (s.r[i] + s.r[j])
            if (dx * dx + dy * dy < lim * lim) parent[find(j)] = find(i)
        }
        val ox = FloatArray(n); val oy = FloatArray(n); val orad = FloatArray(n)
        var m = 0
        for (root in 0 until n) {
            if (find(root) != root) continue
            var sw = 0f; var sx = 0f; var sy = 0f
            for (i in 0 until n) if (find(i) == root) { val wgt = s.r[i] * s.r[i]; sw += wgt; sx += s.x[i] * wgt; sy += s.y[i] * wgt }
            val cx = sx / sw; val cy = sy / sw
            var rad = 0f
            for (i in 0 until n) if (find(i) == root) rad = max(rad, sqrt((s.x[i] - cx) * (s.x[i] - cx) + (s.y[i] - cy) * (s.y[i] - cy)) + s.r[i])
            ox[m] = cx; oy[m] = cy; orad[m] = rad; m++
        }
        // 큰 개체 바로 옆에 붙은 작은 시드는 그 개체의 잘린 가장자리 조각이다. 버린다.
        var out = 0
        for (g in 0 until m) {
            var edgeScrap = false
            for (h in 0 until m) {
                if (h == g || orad[g] >= 0.6f * orad[h]) continue
                val dx = ox[g] - ox[h]; val dy = oy[g] - oy[h]
                if (dx * dx + dy * dy < (1.3f * orad[h]) * (1.3f * orad[h])) { edgeScrap = true; break }
            }
            if (edgeScrap) continue
            s.x[out] = ox[g]; s.y[out] = oy[g]; s.r[out] = orad[g]; out++
        }
        s.n = out
    }

    private fun features(k: Int, seeds: Seeds, w: Int, h: Int): Features? {
        val cx = seeds.x[k]; val cy = seeds.y[k]; val r = seeds.r[k]
        val reach = r * OWN_REACH
        val rIn = r * 0.9f
        val x0 = max(1, (cx - reach).toInt()); val x1 = min(w - 2, (cx + reach).toInt())
        val y0 = max(1, (cy - reach).toInt()); val y1 = min(h - 2, (cy + reach).toInt())

        var cnt = 0
        var sumH = 0f; var sumH2 = 0f; var sumS = 0f; var sumV = 0f
        var green = 0
        var rough = 0f; var roughN = 0
        var sxx = 0f; var syy = 0f; var sxy = 0f
        var disk = 0; var defects = 0
        var bx0 = w; var bx1 = 0; var by0 = h; var by1 = 0

        for (y in y0..y1) for (x in x0..x1) {
            val dx = x - cx; val dy = y - cy
            val d2 = dx * dx + dy * dy
            if (d2 > reach * reach) continue
            if (ownedByOther(k, seeds, x.toFloat(), y.toFloat(), d2)) continue
            val i = y * w + x
            if (mask[i]) {
                cnt++
                bx0 = min(bx0, x); bx1 = max(bx1, x); by0 = min(by0, y); by1 = max(by1, y)
                sumH += hh[i]; sumH2 += hh[i] * hh[i]; sumS += ss[i]; sumV += vv[i]
                if (hh[i] > 45f) green++
                sxx += dx * dx; syy += dy * dy; sxy += dx * dy
                val glare = vv[i] > 0.93f && ss[i] < 0.35f
                if (!glare) {
                    val avg = (vv[i - 1] + vv[i + 1] + vv[i - w] + vv[i + w]) / 4f
                    rough += abs(vv[i] - avg); roughN++
                }
            }
            if (d2 <= rIn * rIn) {
                disk++
                if (!mask[i]) {
                    val dark = vv[i] < DARK_V
                    val whitish = ss[i] < WHITE_S && vv[i] > WHITE_V
                    if (dark || whitish) defects++
                }
            }
        }
        if (cnt < 12 || disk == 0) return null
        // 모양 검사: 원이면 바운딩 박스를 약 0.79 채운다. 사각형(포장재)은 1.0에 가깝다.
        val fill = cnt.toFloat() / ((bx1 - bx0 + 1) * (by1 - by0 + 1))
        if (fill < 0.5f || fill > 0.93f) return null
        if (cnt < 0.45f * PI.toFloat() * r * r || cnt > w * h * 0.25f) return null

        val meanH = sumH / cnt
        val varH = max(0f, sumH2 / cnt - meanH * meanH)
        val a = sxx / cnt; val b = syy / cnt; val c = sxy / cnt
        val tr = a + b
        val det = a * b - c * c
        val disc = sqrt(max(0f, tr * tr / 4 - det))
        val l1 = tr / 2 + disc
        val l2 = max(1e-3f, tr / 2 - disc)
        return Features(
            meanH = meanH,
            stdH = sqrt(varH),
            meanS = sumS / cnt,
            meanV = sumV / cnt,
            greenFrac = green.toFloat() / cnt,
            smooth = if (roughN > 0) rough / roughN else 0f,
            axisRatio = sqrt(l1 / l2),
            defectFrac = defects.toFloat() / disk,
        )
    }

    private val pH = FloatArray(1)
    private val pS = FloatArray(1)
    private val pV = FloatArray(1)

    /**
     * 증거 마스크용 픽셀 분류. 1은 판단에 쓴 과일색, 2는 결함 의심(inner일 때만), 0은 나머지.
     * 화면(UI 스레드)에서만 부른다.
     */
    fun pixelClass(argb: Int, inner: Boolean): Int {
        rgbToHsv(argb, pH, pS, pV, 0)
        if (pH[0] in H_MIN..H_MAX && pS[0] >= S_MIN && pV[0] >= V_MIN) return 1
        if (inner && (pV[0] < DARK_V || (pS[0] < WHITE_S && pV[0] > WHITE_V))) return 2
        return 0
    }

    /** 이 픽셀이 다른 시드에 더 가까우면 그 개체 몫이다. */
    private fun ownedByOther(k: Int, seeds: Seeds, x: Float, y: Float, d2: Float): Boolean {
        for (j in 0 until seeds.n) {
            if (j == k) continue
            val dx = x - seeds.x[j]; val dy = y - seeds.y[j]
            if (dx * dx + dy * dy < d2) return true
        }
        return false
    }
}

/** 0xAARRGGBB를 H(0..360), S, V(0..1)로. */
fun rgbToHsv(argb: Int, h: FloatArray, s: FloatArray, v: FloatArray, i: Int) {
    val r = ((argb shr 16) and 0xFF) / 255f
    val g = ((argb shr 8) and 0xFF) / 255f
    val b = (argb and 0xFF) / 255f
    val mx = max(r, max(g, b))
    val mn = min(r, min(g, b))
    val d = mx - mn
    v[i] = mx
    s[i] = if (mx == 0f) 0f else d / mx
    h[i] = when {
        d == 0f -> 0f
        mx == r -> 60f * (((g - b) / d) % 6f)
        mx == g -> 60f * ((b - r) / d + 2f)
        else -> 60f * ((r - g) / d + 4f)
    }.let { if (it < 0f) it + 360f else it }
}
