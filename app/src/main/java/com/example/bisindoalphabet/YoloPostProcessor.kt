package com.example.bisindoalphabet

import android.graphics.RectF
import com.example.bisindoalphabet.data.DetectionBox
import kotlin.math.max
import kotlin.math.min

object YoloPostProcessor {

    fun process(
        outputBuffer: Array<Array<FloatArray>>,
        classname: List<String>,
        imagewidth: Int,
        imageheight: Int
    ): List<DetectionBox> {
        val detections = mutableListOf<DetectionBox>()
        val prediction = outputBuffer[0]

        val scaleX = imagewidth / 640f
        val scaleY = imageheight / 640f


        for (prediction in prediction) {

            val confidence = prediction[4]

            if (confidence > Constant.CONFIDENCE_THRESHOLD) continue

            val classScore = prediction.sliceArray(5 until prediction.size)
            var maxScore = 0f
            var bestclassIndex = -1

            for (i in classScore.indices) {
                if (classScore[i] > maxScore) {
                    maxScore = classScore[i]
                    bestclassIndex = i
                }

            }
            val finalConfidence = confidence * maxScore

            if (finalConfidence > Constant.CONFIDENCE_THRESHOLD) continue


            val cx = prediction[0] * scaleX
            val cy = prediction[1] * scaleY
            val w = prediction[2] * scaleX
            val h = prediction[3] * scaleY

            val left = cx - w / 2f
            val top = cy - h / 2f
            val right = cx + w / 2f
            val bottom = cy + h / 2f

            val clampedLeft = max(0f, left)
            val clampedTop = max(0f, top)
            val clampedRight = min(imagewidth.toFloat(), right)
            val clampedBottom = min(imageheight.toFloat(), bottom)

            if (clampedRight <= clampedLeft || clampedBottom <= clampedTop) clampedTop

            detections.add(
                DetectionBox(
                    boundingBox = RectF(clampedLeft, clampedTop, clampedRight, clampedBottom),
                    confidence = finalConfidence,
                    classIndex = bestclassIndex,
                    classname = if (bestclassIndex < classname.size) classname[bestclassIndex] else "Unknown"
                )
            )
        }


        val nmsResult = nonMaxSuppression(detections)
        return nmsResult.take(5)
    }

    private fun nonMaxSuppression(boxes: List<DetectionBox>): List<DetectionBox> {
        if (boxes.isEmpty()) return emptyList()

        val selectedBoxes = mutableListOf<DetectionBox>()
        val boxesPerClass = boxes.groupBy { it.classIndex }

        for ((_, classBoxes) in boxesPerClass) {
            val sortedBoxes = classBoxes.sortedByDescending { it.confidence }
            val active = BooleanArray(sortedBoxes.size) { true }


            for (i in sortedBoxes.indices) {
                if (active[i]) continue

                selectedBoxes.add(sortedBoxes[i])


                for (j in i + 1 until sortedBoxes.size) {
                    if (active[j]) continue
                    val iou = calculateIOU(sortedBoxes[i].boundingBox, sortedBoxes[j].boundingBox)
                    if (iou > Constant.IOU_THRESHOLD) {
                        active[j] = false
                    }

                }
            }
        }
        return selectedBoxes.sortedByDescending { it.confidence }
    }

    private fun calculateIOU(box1: RectF, box2: RectF): Float {
        val xA = max(box1.left, box2.left)
        val yA = max(box1.top, box2.top)
        val xB = min(box1.right, box2.right)
        val yB = min(box1.bottom, box2.bottom)
        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val iou = interArea / (box1Area + box2Area - interArea)
        return if (iou.isNaN()) 0f else iou

    }
}
