package com.freshscanner.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import com.freshscanner.app.core.Badge
import com.freshscanner.app.core.Tone
import kotlin.math.hypot
import kotlin.math.max

/** 미리보기 위에 개체별 링과 순위 배지를 그리고, 배지를 누르면 알려준다. 좌표는 정방향 프레임 기준 0..1. */
class OverlayView @JvmOverloads constructor(c: Context, a: AttributeSet? = null) : View(c, a) {
    class Placed(val cx: Float, val cy: Float, val r: Float)

    var onTap: ((Badge) -> Unit)? = null
    private var badges: List<Badge> = emptyList()
    private var frameAspect = 3f / 4f // 정방향 프레임 폭/높이

    private val dp = resources.displayMetrics.density
    private val slop = ViewConfiguration.get(c).scaledTouchSlop
    private val accent = Color.parseColor("#E0661A")
    private val ink = Color.parseColor("#14201A")
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 14f * dp; isFakeBoldText = true; textAlign = Paint.Align.CENTER
    }
    private var downX = 0f
    private var downY = 0f

    fun show(b: List<Badge>, aspect: Float) {
        badges = b
        frameAspect = aspect
        invalidate()
    }

    /** PreviewView(FILL_CENTER)와 같이 화면을 꽉 채우고 넘치는 부분은 잘라낸다. */
    fun place(b: Badge, aspect: Float = frameAspect): Placed {
        val (ox, oy, fw, fh) = cover(aspect)
        return Placed(ox + b.x * fw, oy + b.y * fh, b.r * fw)
    }

    /** 정규화 좌표 (x, y)가 화면 안에 보이는지. 잘려 나간 개체는 순위 비교에서도 뺀다. */
    fun isVisible(x: Float, y: Float, aspect: Float): Boolean {
        if (width == 0 || height == 0) return true
        val (ox, oy, fw, fh) = cover(aspect)
        return ox + x * fw in 0f..width.toFloat() && oy + y * fh in 0f..height.toFloat()
    }

    /** 프레임이 화면을 덮는 사각형: 왼쪽 위 x, y와 폭, 높이. 넘치는 부분은 음수 오프셋으로 잘린다. */
    private fun cover(aspect: Float): FloatArray {
        val vw = width.toFloat(); val vh = height.toFloat()
        val fw: Float; val fh: Float
        if (vw / vh > aspect) { fw = vw; fh = vw / aspect } else { fh = vh; fw = vh * aspect }
        return floatArrayOf((vw - fw) / 2f, (vh - fh) / 2f, fw, fh)
    }

    private operator fun FloatArray.component4() = this[3]

    override fun onDraw(canvas: Canvas) {
        for (b in badges) {
            val p = place(b)
            val rr = max(p.r, 14f * dp)
            val best = b.tone == Tone.BEST
            ring.color = if (best) accent else Color.argb(220, 255, 255, 255)
            ring.strokeWidth = (if (best) 4f else 2.5f) * dp
            canvas.drawCircle(p.cx, p.cy, rr, ring)

            val (bg, fg) = when (b.tone) {
                Tone.BEST -> accent to ink
                Tone.OK -> Color.WHITE to ink
                Tone.WARN -> ink to Color.WHITE
            }
            val box = pill(b.label, p.cx, p.cy, rr)
            fill.color = bg
            canvas.drawRoundRect(box, box.height() / 2, box.height() / 2, fill)
            text.color = fg
            canvas.drawText(b.label, p.cx, box.centerY() - (text.ascent() + text.descent()) / 2, text)
        }
    }

    private fun pill(label: String, cx: Float, cy: Float, rr: Float): RectF {
        val tw = text.measureText(label)
        val padX = 10f * dp
        val h = 26f * dp
        val top = cy + rr + 6f * dp
        var l = cx - tw / 2 - padX
        var r = cx + tw / 2 + padX
        val m = 8f * dp // 화면 가장자리에서 배지가 잘리지 않게 안쪽으로 민다
        if (l < m) { r += m - l; l = m } else if (r > width - m) { l -= r - (width - m); r = width - m }
        return RectF(l, top, r, top + h)
    }

    override fun onTouchEvent(e: MotionEvent): Boolean {
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> { downX = e.x; downY = e.y }
            MotionEvent.ACTION_UP -> if (hypot(e.x - downX, e.y - downY) < slop) {
                val hit = hitTest(e.x, e.y)
                if (hit != null) { performClick(); onTap?.invoke(hit) }
            }
        }
        return true
    }

    override fun performClick(): Boolean { super.performClick(); return true }

    /** 링 안쪽, 또는 배지(세로 48dp까지 넓힌 영역)를 누르면 맞은 것으로 본다. 여럿이면 중심이 가까운 쪽. */
    private fun hitTest(x: Float, y: Float): Badge? {
        var best: Badge? = null
        var bestD = Float.MAX_VALUE
        for (b in badges) {
            val p = place(b)
            val rr = max(p.r, 14f * dp)
            val box = pill(b.label, p.cx, p.cy, rr).apply { inset(0f, -11f * dp) }
            val d = hypot(x - p.cx, y - p.cy)
            if ((d <= rr + 8f * dp || box.contains(x, y)) && d < bestD) { best = b; bestD = d }
        }
        return best
    }
}
