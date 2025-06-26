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

        for (prediction in prediction) {

            val confidence = prediction[4]

            if (confidence > Constant.CONFIDENCE_THRESHOLD) {
                val classScore = prediction.sliceArray(5 until prediction.size)
                var maxScore = 0f
                var classIndex = -1

                for (i in classScore.indices) {
                    if (classScore[i] > maxScore) {
                        maxScore = classScore[i]
                        classIndex = i
                    }

                }
                val finalConfidence = confidence * maxScore

                if (finalConfidence > Constant.CONFIDENCE_THRESHOLD) {
                    val scaleX = imagewidth / 640f
                    val scaleY = imageheight / 640f

                    val cx = prediction[0] * scaleX
                    val cy = prediction[1] * scaleY
                    val w = prediction[2] * scaleX
                    val h = prediction[3] * scaleY

                    val left = cx - w / 2
                    val top = cy - h / 2
                    val right = cx + w / 2
                    val bottom = cy + h / 2

                    detections.add(
                        DetectionBox(
                            boundingBox = RectF(left, top, right, bottom),
                            confidence = confidence * maxScore,
                            classIndex = classIndex,
                            classname = if (classIndex < classname.size) classname[classIndex] else "Unknown"
                        )
                    )
                }

            }
        }
        val nmsResult = nonMaxSuppression(detections)
        return if(nmsResult.isEmpty()){
            val bestResult = nmsResult.maxByOrNull { it.confidence }
            listOfNotNull(bestResult)
        }else{
            emptyList()
        }
    }
    private fun nonMaxSuppression(boxes: List<DetectionBox>): List<DetectionBox> {
        val selectedBoxes = mutableListOf<DetectionBox>()
        val boxesPerClass = boxes.groupBy { it.classIndex }

        for ((_, classBoxes) in boxesPerClass) {
            val sortedBoxes = classBoxes.sortedByDescending { it.confidence }
            val active = BooleanArray(sortedBoxes.size){true}
            var numActive = active.size


            for (i in sortedBoxes.indices){
                if (active[i]){
                    selectedBoxes.add(sortedBoxes[i])
                    if (numActive ==  1 )break

                    for (j in i+1 until sortedBoxes.size){
                        if(active[j]){
                            val iou =  calculateIOU(sortedBoxes[i].boundingBox, sortedBoxes[j].boundingBox)
                            if (iou > Constant.IOU_THRESHOLD){
                                active [j] = false
                                numActive--
                            }
                        }
                    }
                }
            }
        }
        return selectedBoxes
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
