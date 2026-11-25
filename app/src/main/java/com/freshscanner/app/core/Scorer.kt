package com.freshscanner.app.core

import kotlin.math.abs

/**
 * 감귤 룰 점수 (설계: 프레시스캐너-검증/실측설계.md의 룰 4항목 중 화면에서 계산 가능한 3개).
 * 꼭지는 v0에서 계산하지 않는다.
 * 아래 임계값은 통설 기반 가설이다. 실측(Brix)으로 보정하기 전까지 정확도를 주장하지 않는다.
 */
object Scorer {
    const val VETO_DEFECT = 0.03f // 가설: 겉면의 3%쯤이면 눈에 띄는 반점이다. 실측으로 보정한다

    /** color, texture, flat은 각각 0..2. 화면의 요인 막대가 이 값을 쓴다. */
    data class Result(
        val score: Float,
        val veto: Boolean,
        val reason: String,
        val color: Float,
        val texture: Float,
        val flat: Float,
    )

    fun score(f: Features): Result {
        val hueFit = clamp01(1f - abs(f.meanH - 24f) / 18f)
        val uniform = clamp01(1f - f.stdH / 10f)
        var color = 2f * hueFit * (0.6f + 0.4f * uniform)
        if (f.greenFrac > 0.15f) color *= 0.7f // 연두빛이 많으면 덜 익은 쪽

        val texture = 2f * clamp01(1f - f.smooth / 0.10f)

        val q = f.axisRatio
        val flat = when {
            q < 1.08f -> 1f + clamp01((q - 1f) / 0.08f)
            q <= 1.40f -> 2f
            else -> (2f - (q - 1.4f) / 0.3f).coerceAtLeast(0f)
        }

        val raw = 100f * (0.45f * color / 2f + 0.30f * texture / 2f + 0.25f * flat / 2f)
        val veto = f.defectFrac > VETO_DEFECT
        val score = if (veto) minOf(raw, 35f) else raw

        val reason = when {
            veto -> "어두운 반점이 보여요"
            color / 2f >= 0.8f && texture / 2f >= 0.6f -> "색이 고르고 껍질이 매끈해요"
            color / 2f >= 0.8f -> "색이 고르고 진해요"
            texture / 2f < 0.4f -> "껍질 결이 거칠어 보여요"
            f.greenFrac > 0.15f -> "연두빛이 많아 덜 익어 보여요"
            else -> "색이 조금 연해요"
        }
        return Result(score, veto, reason, color, texture, flat)
    }

    private fun clamp01(x: Float) = x.coerceIn(0f, 1f)
}
