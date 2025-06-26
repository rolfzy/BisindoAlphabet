package com.example.bisindoalphabet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.bisindoalphabet.data.DetectionBox
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.concurrent.Executors

data class TfliteContainer(val interpreter: Interpreter, val gpuDelegate: GpuDelegate?)

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun LiveDetection() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)

    var detectionBoxes by remember { mutableStateOf(emptyList<DetectionBox>()) }
    var inferenceTime by remember { mutableStateOf(0L) }
    var imageSize by remember { mutableStateOf(Size(0, 0)) }

    // PERBAIKAN 1: Inisialisasi TFLite menggunakan '=' bukan 'by'
    val tflite = remember { setupTflite(context) }
    val classNames = remember {
        FileUtil.loadLabels(
            context,
            Constant.LABELS_PATH
        )
    } // PERBAIKAN 2: Nama konstanta

    DisposableEffect(Unit) {
        onDispose {
            tflite.interpreter.close()
            tflite.gpuDelegate?.close()
            Log.i("LiveDetectionPage", "TFLite resources released.")
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (cameraPermissionState.status.isGranted) {
            CameraView(
                onBitmapReady = { bitmap ->
                    // Simpan ukuran bitmap untuk digunakan di overlay
                    imageSize = Size(bitmap.width, bitmap.height)
                    val startTime = System.currentTimeMillis()
                    // Pastikan YoloPostProcessor ada dan benar
                    val results = runInference(bitmap, tflite.interpreter, classNames)
                    val endTime = System.currentTimeMillis()

                    detectionBoxes = results
                    inferenceTime = endTime - startTime
                },
                lifecycleOwner = lifecycleOwner,
                context = context
            )

            DetectionOverlay(
                boxes = detectionBoxes,
                sourceImageWidth = imageSize.width,
                sourceImageHeight = imageSize.height
            )

            Text(
                text = "Time: $inferenceTime ms",
                color = Color.White,
                modifier = Modifier.align(Alignment.TopCenter)
            )
        } else {
            LaunchedEffect(Unit) {
                cameraPermissionState.launchPermissionRequest()
            }
            Text("Camera Permission Not Granted", modifier = Modifier.align(Alignment.Center))
        }
    }
}

@Composable
fun CameraView(
    onBitmapReady: (Bitmap) -> Unit,
    lifecycleOwner: LifecycleOwner,
    context: Context
) {
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                this.scaleType = PreviewView.ScaleType.FILL_CENTER
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                // PERBAIKAN 3: Bangun Preview terlebih dahulu, baru set SurfaceProvider
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 640)) // Sesuai input model
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val bitmap = imageProxy.toBitmap()

                            val rotatedBitmap = if (rotationDegrees != 0) {
                                val matrix =
                                    Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                                Bitmap.createBitmap(
                                    bitmap,
                                    0,
                                    0,
                                    bitmap.width,
                                    bitmap.height,
                                    matrix,
                                    true
                                )
                            } else {
                                bitmap
                            }
                            onBitmapReady(rotatedBitmap)
                            imageProxy.close()
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    // PERBAIKAN 4: bindToLifecycle sekarang menerima argumen yang benar
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, // Kirim objek Preview, bukan builder
                        imageAnalyzer
                    )
                } catch (e: Exception) {
                    Log.e("CameraView", "Use case binding failed", e)
                }
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        }
    )
}

@Composable
fun DetectionOverlay(boxes: List<DetectionBox>, sourceImageWidth: Int, sourceImageHeight: Int) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        if (sourceImageWidth == 0 || sourceImageHeight == 0) return@Canvas

        // PERBAIKAN 5: Logika skala yang benar
        val scaleX = size.width / sourceImageWidth
        val scaleY = size.height / sourceImageHeight

        boxes.forEach { box ->
            val rect = box.boundingBox
            // Koordinat BBox sudah diskalakan ke ukuran bitmap oleh YoloPostProcessor,
            // sekarang kita hanya perlu menskalakannya ke ukuran canvas.
            val left = rect.left * scaleX
            val top = rect.top * scaleY
            val right = rect.right * scaleX
            val bottom = rect.bottom * scaleY

            drawRect(
                color = Color.Red,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = 4f)
            )
        }
    }
}

private fun setupTflite(context: Context): TfliteContainer {
    val modelBuffer = FileUtil.loadMappedFile(context, Constant.MODEL_PATH)
    var gpuDelegate: GpuDelegate?
    // PERBAIKAN 6: Deklarasikan 'interpreter' dengan 'val'
    var interpreter: Interpreter

    try {
        gpuDelegate = GpuDelegate()
        val options = Interpreter.Options().addDelegate(gpuDelegate)
        interpreter = Interpreter(modelBuffer, options)
        Log.i("TFLiteSetup", "GPU delegate enabled.")
    } catch (e: Exception) {
        gpuDelegate = null
        interpreter = Interpreter(modelBuffer) // Fallback to CPU
        Log.e("TFLiteSetup", "GPU Delegate failed to initialize, falling back to CPU.", e)
    }
    return TfliteContainer(interpreter, gpuDelegate)
}
private fun runInference(
    bitmap: Bitmap,
    interpreter: Interpreter,
    classNames: List<String>
): List<DetectionBox> {
    // Dimensi input model
    val modelInputWidth = 640
    val modelInputHeight = 640

    // ================== PERBAIKAN 1: KEMBALI MENGGUNAKAN UINT8 ==================
    // TensorImage HANYA menerima UINT8 atau FLOAT32. Kita gunakan UINT8 untuk model kuantisasi.
    val tensorImage = TensorImage(org.tensorflow.lite.DataType.UINT8)
    tensorImage.load(bitmap)

    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    val processedImage = imageProcessor.process(tensorImage)
    val inputBuffer = processedImage.buffer
    // =========================================================================

    // Dapatkan detail tensor output
    val outputTensor = interpreter.getOutputTensor(0)
    val outputShape = outputTensor.shape()
    val numPredictions = outputShape[2] // 8400
    val numAttributes = outputShape[1] // 30

    // Buat buffer output untuk menerima data mentah INT8
    val rawOutputBuffer = Array(1) { Array(numAttributes) { ByteArray(numPredictions) } }

    // Jalankan inferensi
    interpreter.run(inputBuffer, rawOutputBuffer)

    // De-kuantisasi dan Transpose
    val quantizationParams = outputTensor.quantizationParams()
    val scale = quantizationParams.scale
    val zeroPoint = quantizationParams.zeroPoint

    val dequantizedOutput = Array(1) { Array(numPredictions) { FloatArray(numAttributes) } }

    // ================== PERBAIKAN 2: PASTIKAN LOOP INI BENAR ==================
    // Ini adalah sumber error 'ArrayIndexOutOfBoundsException'.
    // Pastikan Anda menggunakan 'until' yang berarti "berjalan sampai SEBELUM".

    // 'i' berjalan dari 0 sampai 8399 (total 8400 kali)
    for (i in 0 until numPredictions) {
        // 'j' berjalan dari 0 sampai 29 (total 30 kali)
        for (j in 0 until numAttributes) { // PASTIKAN KATA 'until'

            val intVal = rawOutputBuffer[0][j][i].toInt() and 0xFF
            val floatVal = scale * (intVal - zeroPoint)
            dequantizedOutput[0][i][j] = floatVal
        }
    }

    return YoloPostProcessor.process(
        dequantizedOutput,
        classNames,
        bitmap.width,
        bitmap.height
    )
}