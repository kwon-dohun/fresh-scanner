package com.freshscanner.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.drawToBitmap
import androidx.core.view.WindowInsetsCompat
import com.freshscanner.app.core.Badge
import com.freshscanner.app.core.Frame
import com.freshscanner.app.core.FruitDetector
import com.freshscanner.app.core.Scorer
import com.freshscanner.app.core.Tone
import com.freshscanner.app.core.Tracker
import com.freshscanner.app.core.downsampleUpright
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var summary: TextView
    private lateinit var scrim: View
    private lateinit var sheet: View
    private lateinit var evidence: EvidenceView
    private lateinit var evidenceCaption: View
    private lateinit var permission: View

    private val tracker = Tracker()
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var latest: List<Badge> = emptyList()      // 화면 안에 보이는 개체 (UI 스레드)
    @Volatile private var lastFrame: Frame? = null
    private var frameCount = 0                          // 분석 스레드에서만 쓴다
    private var cameraStarted = false
    private var demoView: ImageView? = null

    private val closeSheetOnBack = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() = hideSheet()
    }

    private val askCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
            if (ok) startCamera() else permission.visibility = View.VISIBLE
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        preview = findViewById(R.id.preview)
        preview.scaleType = PreviewView.ScaleType.FILL_CENTER
        overlay = findViewById(R.id.overlay)
        summary = findViewById(R.id.summary)
        scrim = findViewById(R.id.scrim)
        sheet = findViewById(R.id.sheet)
        evidence = findViewById(R.id.evidence)
        evidenceCaption = findViewById(R.id.evidenceCaption)
        permission = findViewById(R.id.permission)

        // 상태바와 내비게이션 바 영역을 피해서 상단 바, 요약 알약, 시트 여백을 잡는다.
        val topBar = findViewById<TextView>(R.id.topBar)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            topBar.setPadding(dp(20), bars.top + dp(14), dp(20), dp(14))
            (summary.layoutParams as FrameLayout.LayoutParams).bottomMargin = bars.bottom + dp(16)
            summary.requestLayout()
            sheet.setPadding(dp(20), dp(20), dp(20), bars.bottom + dp(20))
            insets
        }

        overlay.onTap = ::showSheet
        scrim.setOnClickListener { hideSheet() }
        findViewById<View>(R.id.sheetClose).setOnClickListener { hideSheet() }
        findViewById<View>(R.id.sheetSave).setOnClickListener { saveCapture() }
        findViewById<Button>(R.id.permissionButton).setOnClickListener { requestOrOpenSettings() }
        onBackPressedDispatcher.addCallback(this, closeSheetOnBack)

        if (Demo.requested(this)) startDemo()
        else if (hasCamera()) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    /** 디버그 전용 데모: 카메라 대신 합성 사진을 같은 파이프라인에 넣는다. */
    private fun startDemo() {
        cameraStarted = true
        preview.visibility = View.GONE
        val iv = findViewById<ImageView>(R.id.demoView)
        demoView = iv
        val bmp = Demo.bitmap(768, 1024)
        iv.setImageBitmap(bmp)
        iv.visibility = View.VISIBLE
        val small = Bitmap.createScaledBitmap(bmp, FRAME_W, FRAME_W * 4 / 3, true)
        val px = IntArray(small.width * small.height)
        small.getPixels(px, 0, small.width, 0, 0, small.width, small.height)
        val frame = Frame(small.width, small.height, px)
        lastFrame = frame
        overlay.post { // 오버레이 크기가 정해진 뒤에 화면 안 과일만 골라 순위를 매긴다
            val aspect = frame.w.toFloat() / frame.h
            val dets = FruitDetector.detect(frame).filter { d -> overlay.isVisible(d.x, d.y, aspect) }
            var badges: List<Badge> = emptyList()
            repeat(6) { badges = tracker.update(dets) } // 점수 이동평균이 자리잡게
            latest = badges
            overlay.show(badges, aspect)
            summary.text = describe(badges)
        }
    }

    /** 오버레이와 같은 좌표계의 화면 이미지. 데모 모드면 데모 사진을 쓴다. */
    private fun screenBitmap(): Bitmap? =
        demoView?.takeIf { it.visibility == View.VISIBLE }?.drawToBitmap() ?: preview.bitmap

    override fun onResume() {
        super.onResume()
        // 설정에서 권한을 켜고 돌아온 경우
        if (!cameraStarted && hasCamera()) startCamera()
    }

    private fun hasCamera() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestOrOpenSettings() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
            askCamera.launch(Manifest.permission.CAMERA)
        } else {
            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)))
        }
    }

    private fun startCamera() {
        cameraStarted = true
        permission.visibility = View.GONE
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .build()
            val prev = Preview.Builder().setResolutionSelector(selector).build()
                .also { it.surfaceProvider = preview.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, prev, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(img: ImageProxy) {
        img.use {
            val t0 = System.nanoTime()
            val plane = it.planes[0]
            val frame = downsampleUpright(
                plane.buffer, plane.rowStride, plane.pixelStride,
                it.width, it.height, it.imageInfo.rotationDegrees, FRAME_W,
            )
            val aspect = frame.w.toFloat() / frame.h
            // 화면 밖으로 잘려 나간 과일은 순위 비교에서도 뺀다. 안 그러면 보이지 않는 1위가 생긴다.
            val dets = FruitDetector.detect(frame).filter { d -> overlay.isVisible(d.x, d.y, aspect) }
            val badges = tracker.update(dets)
            if (++frameCount % 30 == 0) {
                Log.d("FS", "분석 ${(System.nanoTime() - t0) / 1_000_000} ms (원본 ${it.width}x${it.height})")
            }
            lastFrame = frame
            runOnUiThread {
                latest = badges
                overlay.show(badges, aspect)
                summary.text = describe(badges)
            }
        }
    }

    /** 요약은 한 줄 결론만. 이유는 배지를 눌렀을 때 시트에서 보여준다. */
    private fun describe(b: List<Badge>): String {
        if (b.isEmpty()) return "감귤을 비춰 보세요"
        if (b.size == 1) return "하나 더 비추면 순위를 보여드려요"
        val best = b.firstOrNull { it.tone == Tone.BEST } ?: return "주의가 필요한 감귤이 많아요"
        return "${where(best)}이 1위예요"
    }

    private fun where(b: Badge): String {
        val h = if (b.x < 0.38f) "왼쪽" else if (b.x > 0.62f) "오른쪽" else "가운데"
        val v = if (b.y < 0.38f) "위" else if (b.y > 0.62f) "아래" else ""
        return "$h $v".trim() + " 감귤"
    }

    private fun showSheet(b: Badge) {
        val frame = lastFrame ?: return
        val res = Scorer.score(b.features)
        findViewById<TextView>(R.id.sheetRank).text = b.label
        findViewById<TextView>(R.id.sheetScore).text = "종합 ${b.score.roundToInt()}점"
        findViewById<TextView>(R.id.sheetReason).text = b.reason
        findViewById<ProgressBar>(R.id.barColor).progress = (res.color / 2f * 100).roundToInt()
        findViewById<ProgressBar>(R.id.barTexture).progress = (res.texture / 2f * 100).roundToInt()
        findViewById<ProgressBar>(R.id.barShape).progress = (res.flat / 2f * 100).roundToInt()

        val screen = screenBitmap()
        if (screen != null) {
            val p = overlay.place(b)
            val (photo, mask) = Evidence.build(screen, frame, b, latest, p.cx, p.cy, p.r)
            evidence.set(photo, mask)
            evidence.visibility = View.VISIBLE
            evidenceCaption.visibility = View.VISIBLE
        } else {
            evidence.visibility = View.GONE
            evidenceCaption.visibility = View.GONE
        }

        scrim.alpha = 0f
        scrim.visibility = View.VISIBLE
        scrim.animate().alpha(1f).setDuration(200).start()
        sheet.visibility = View.VISIBLE
        sheet.post {
            sheet.translationY = sheet.height.toFloat()
            sheet.animate().translationY(0f).setDuration(240).setInterpolator(DecelerateInterpolator()).start()
        }
        closeSheetOnBack.isEnabled = true
    }

    private fun hideSheet() {
        closeSheetOnBack.isEnabled = false
        scrim.animate().alpha(0f).setDuration(160).withEndAction { scrim.visibility = View.GONE }.start()
        sheet.animate().translationY(sheet.height.toFloat()).setDuration(180)
            .withEndAction { sheet.visibility = View.GONE }.start()
    }

    /** 실측 기록용: 현재 화면과 개체별 원시 특징을 저장한다. Brix는 나중에 직접 채운다. */
    private fun saveCapture() {
        val badges = latest
        if (badges.isEmpty()) {
            Toast.makeText(this, "저장할 감귤이 화면에 없어요", Toast.LENGTH_SHORT).show()
            return
        }
        val bmp: Bitmap = screenBitmap() ?: run {
            Toast.makeText(this, "화면을 아직 읽지 못했어요", Toast.LENGTH_SHORT).show()
            return
        }
        val dir = getExternalFilesDir("captures") ?: filesDir
        val name = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        File(dir, "$name.jpg").outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }

        val items = JSONArray()
        for (b in badges) {
            items.put(JSONObject().apply {
                put("id", b.id); put("x", b.x); put("y", b.y); put("r", b.r)
                put("label", b.label); put("score", b.score)
                put("meanH", b.features.meanH); put("stdH", b.features.stdH)
                put("meanS", b.features.meanS); put("meanV", b.features.meanV)
                put("greenFrac", b.features.greenFrac); put("smooth", b.features.smooth)
                put("axisRatio", b.features.axisRatio); put("defectFrac", b.features.defectFrac)
                put("brix", JSONObject.NULL) // 실측 후 직접 입력
            })
        }
        File(dir, "$name.json").writeText(JSONObject().put("fruits", items).toString(2))
        Toast.makeText(this, "기록했어요: $name", Toast.LENGTH_SHORT).show()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private companion object {
        const val FRAME_W = 256
    }
}
