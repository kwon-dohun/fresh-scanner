package com.freshscanner.app.core

import kotlin.math.hypot

enum class Tone { BEST, OK, WARN }

/** 화면에 그릴 한 개체. 순위는 이번 프레임에 보이는 것들끼리의 상대 비교다. */
data class Badge(
    val id: Int,
    val x: Float,
    val y: Float,
    val r: Float,
    val label: String,
    val tone: Tone,
    val score: Float,
    val reason: String,
    val features: Features,
)

/** 프레임마다 흔들리는 점수를 지수이동평균으로 안정화하고, 같은 개체는 같은 id로 이어 붙인다. */
class Tracker {
    private class Track(var id: Int, var x: Float, var y: Float, var r: Float, var score: Float) {
        var missed = 0
        var veto = false
        var reason = ""
        lateinit var features: Features
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1

    fun update(dets: List<Detection>): List<Badge> {
        val used = BooleanArray(tracks.size)
        for (d in dets) {
            val res = Scorer.score(d.features)
            var best = -1
            var bestDist = 0.08f // 프레임 폭 대비 거리
            for (i in used.indices) { // 이번 프레임에 새로 생긴 트랙은 매칭 대상이 아니다
                if (used[i]) continue
                val dist = hypot(tracks[i].x - d.x, tracks[i].y - d.y)
                if (dist < bestDist) { bestDist = dist; best = i }
            }
            if (best >= 0) {
                val t = tracks[best]
                used[best] = true
                t.x = d.x; t.y = d.y; t.r = d.r
                t.score += ALPHA * (res.score - t.score)
                t.veto = res.veto; t.reason = res.reason; t.features = d.features
                t.missed = 0
            } else {
                val t = Track(nextId++, d.x, d.y, d.r, res.score)
                t.veto = res.veto; t.reason = res.reason; t.features = d.features
                tracks.add(t)
            }
        }
        // 이번 프레임에 못 본 트랙은 잠깐 유지했다가 버린다.
        for (i in used.indices) if (!used[i]) tracks[i].missed++
        tracks.removeAll { it.missed > MAX_MISSED }

        val visible = tracks.filter { it.missed == 0 }
        val ranked = visible.sortedWith(compareBy<Track> { it.veto }.thenByDescending { it.score })
        var rank = 0
        return ranked.map { t ->
            if (t.veto) {
                Badge(t.id, t.x, t.y, t.r, "주의", Tone.WARN, t.score, t.reason, t.features)
            } else {
                rank++
                val tone = if (rank == 1 && visible.size >= 2) Tone.BEST else Tone.OK
                val label = if (tone == Tone.BEST) "1위 추천" else if (visible.size == 1) "후보" else "${rank}위"
                Badge(t.id, t.x, t.y, t.r, label, tone, t.score, t.reason, t.features)
            }
        }
    }

    private companion object {
        const val ALPHA = 0.35f
        const val MAX_MISSED = 5
    }
}
