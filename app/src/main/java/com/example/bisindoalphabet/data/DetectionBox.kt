package com.example.bisindoalphabet.data

import android.graphics.RectF

data class DetectionBox (
    val boundingBox: RectF,
    val confidence: Float,
    val classIndex: Int,
    val classname: String
)