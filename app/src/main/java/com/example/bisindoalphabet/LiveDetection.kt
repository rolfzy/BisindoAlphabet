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
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
import java.util.concurrent.atomic.AtomicBoolean

data class TfliteContainer(val interpreter: Interpreter, val gpuDelegate: GpuDelegate?)

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun LiveDetection() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)

    var detectionBoxes by remember { mutableStateOf(emptyList<DetectionBox>()) }
    var inferenceTime by remember { mutableStateOf(0L) }
    var fps by remember { mutableStateOf(0f) }

    var imageSize by remember { mutableStateOf(Size(0, 0)) }
    var isProcessing by remember { mutableStateOf(false) }

    val tflite = remember { setupTflite(context) }
    val classNames = remember {
        FileUtil.loadLabels(
            context,
            Constant.LABELS_PATH
        )
    }

    var frameCount by remember { mutableStateOf(0) }
    var lastFpsTime by remember { mutableStateOf(System.currentTimeMillis()) }

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

                    frameCount++
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFpsTime >= 1000) {
                        fps = frameCount * 1000f / (currentTime - lastFpsTime)
                        frameCount = 0
                        lastFpsTime = currentTime
                    }
                    isProcessing = false

                },
                lifecycleOwner = lifecycleOwner,
                context = context
            )

            DetectionOverlay(
                boxes = detectionBoxes,
                sourceImageWidth = imageSize.width,
                sourceImageHeight = imageSize.height
            )

            Card(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
            ){
                Text(
                    text = "FPS: ${"%.1f".format(fps)}\nInference: ${inferenceTime}ms\nDetections: ${detectionBoxes.size}",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(8.dp)
                )
            }


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
    val isProcessing = remember { AtomicBoolean(false) }

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


                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 640))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->

                            if(isProcessing.compareAndSet(false,true)) {
                                try {
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
                                } finally {
                                    isProcessing.set(false)
                                    imageProxy.close()
                                }
                            }else{
                                imageProxy.close()
                            }
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
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

        val scaleX = size.width / sourceImageWidth
        val scaleY = size.height / sourceImageHeight

        boxes.forEach { box ->
            val rect = box.boundingBox
            val left = rect.left * scaleX
            val top = rect.top * scaleY
            val right = rect.right * scaleX
            val bottom = rect.bottom * scaleY

            drawRect(
                color = Color.Red,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = 3f)
            )

            drawRect(
                color = Color.Red,
                topLeft = Offset(left,top - 30f),
                size = androidx.compose.ui.geometry.Size(
                    (box.classname.length * 12f).coerceAtLeast(80f),
                    25f
                )
            )
        }
    }
}
private fun runInference(
    bitmap: Bitmap,
    interpreter: Interpreter,
    classNames: List<String>
): List<DetectionBox> {
    // Dimensi input model
    val modelInputWidth = 640
    val modelInputHeight = 640

    // PERBAIKAN UTAMA: Periksa format input tensor yang diharapkan model
    val inputTensor = interpreter.getInputTensor(0)
    val inputShape = inputTensor.shape()
    val inputDataType = inputTensor.dataType()

    Log.d("TensorInfo", "Input shape: ${inputShape.contentToString()}")
    Log.d("TensorInfo", "Input data type: $inputDataType")
    Log.d("TensorInfo", "Expected bytes: ${inputTensor.numBytes()}")

    // Gunakan TensorImage dengan data type yang sesuai dengan model
    val tensorImage = when (inputDataType) {
        DataType.FLOAT32 -> TensorImage(DataType.FLOAT32)
        DataType.UINT8 -> TensorImage(DataType.UINT8)
        else -> {
            Log.e("TensorError", "Unsupported input data type: $inputDataType")
            TensorImage(DataType.FLOAT32) // fallback
        }
    }

    tensorImage.load(bitmap)

    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    val processedImage = imageProcessor.process(tensorImage)
    val inputBuffer = processedImage.buffer

    // Log ukuran buffer untuk debugging
    Log.d("BufferInfo", "Input buffer size: ${inputBuffer.remaining()} bytes")

    // Dapatkan detail tensor output
    val outputTensor = interpreter.getOutputTensor(0)
    val outputShape = outputTensor.shape()
    val outputDataType = outputTensor.dataType()

    Log.d("OutputTensorInfo", "Output shape: ${outputShape.contentToString()}")
    Log.d("OutputTensorInfo", "Output data type: $outputDataType")

    // Buat buffer output berdasarkan tipe data
    val outputBuffer = when (outputDataType) {
        DataType.FLOAT32 -> {
            val numPredictions = outputShape[2] // 8400
            val numAttributes = outputShape[1] // 30
            Array(1) { Array(numAttributes) { FloatArray(numPredictions) } }
        }
        DataType.UINT8 -> {
            val numPredictions = outputShape[2] // 8400
            val numAttributes = outputShape[1] // 30
            Array(1) { Array(numAttributes) { ByteArray(numPredictions) } }
        }
        else -> {
            Log.e("TensorError", "Unsupported output data type: $outputDataType")
            return emptyList()
        }
    }

    try {
        // Jalankan inferensi
        interpreter.run(inputBuffer, outputBuffer)
    } catch (e: Exception) {
        Log.e("InferenceError", "Error during inference", e)
        return emptyList()
    }

    // Proses output berdasarkan tipe data
    return when (outputDataType) {
        DataType.FLOAT32 -> {
            @Suppress("UNCHECKED_CAST")
            val floatOutput = outputBuffer as Array<Array<FloatArray>>
            processFloat32Output(floatOutput, classNames, bitmap.width, bitmap.height)
        }
        DataType.UINT8 -> {
            @Suppress("UNCHECKED_CAST")
            val byteOutput = outputBuffer as Array<Array<ByteArray>>
            processUint8Output(byteOutput, outputTensor, classNames, bitmap.width, bitmap.height)
        }
        else -> emptyList()
    }
}

private fun processFloat32Output(
    outputBuffer: Array<Array<FloatArray>>,
    classNames: List<String>,
    imageWidth: Int,
    imageHeight: Int
): List<DetectionBox> {
    val numPredictions = outputBuffer[0][0].size // 8400
    val numAttributes = outputBuffer[0].size // 30
    val transposedOutput = Array(1) { Array(numPredictions) { FloatArray(numAttributes) } }

    for (i in 0 until numPredictions) {
        for (j in 0 until numAttributes) {
            transposedOutput[0][i][j] = outputBuffer[0][j][i]
        }
    }

    return YoloPostProcessor.process(transposedOutput, classNames, imageWidth, imageHeight)
}

private fun processUint8Output(
    rawOutputBuffer: Array<Array<ByteArray>>,
    outputTensor: org.tensorflow.lite.Tensor,
    classNames: List<String>,
    imageWidth: Int,
    imageHeight: Int
): List<DetectionBox> {
    val numPredictions = rawOutputBuffer[0][0].size // 8400
    val numAttributes = rawOutputBuffer[0].size // 30

    // De-kuantisasi
    val quantizationParams = outputTensor.quantizationParams()
    val scale = quantizationParams.scale
    val zeroPoint = quantizationParams.zeroPoint

    val dequantizedOutput = Array(1) { Array(numPredictions) { FloatArray(numAttributes) } }

    for (i in 0 until numPredictions) {
        for (j in 0 until numAttributes) {
            val intVal = rawOutputBuffer[0][j][i].toInt() and 0xFF
            val floatVal = scale * (intVal - zeroPoint)
            dequantizedOutput[0][i][j] = floatVal
        }
    }

    return YoloPostProcessor.process(dequantizedOutput, classNames, imageWidth, imageHeight)
}


private fun setupTflite(context: Context): TfliteContainer {
    val modelBuffer = FileUtil.loadMappedFile(context, Constant.MODEL_PATH)
    var gpuDelegate: GpuDelegate? = null
    var interpreter: Interpreter

    try {
        gpuDelegate = GpuDelegate()
        val options = Interpreter.Options().apply {
            addDelegate(gpuDelegate)
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)
        Log.i("TFLiteSetup", "GPU delegate enabled with 4 threads.")
    } catch (e: Exception) {
        gpuDelegate?.close()
        gpuDelegate = null
        val options  = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        interpreter = Interpreter(modelBuffer)
        Log.i("TFLiteSetup", "CPU mode with NNAPI enabled.")

    }
    return TfliteContainer(interpreter, gpuDelegate)
}
