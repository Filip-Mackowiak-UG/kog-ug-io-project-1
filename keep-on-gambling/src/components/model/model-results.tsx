"use client";

import { useImageDetectionContext } from "@/providers/image-detection-provider";
import Image from "next/image";
import React from "react";

interface ServerPrediction {
  imgData?: string;
  predictedClass?: string;
  cropped?: string;
  error?: string;
  detected?: string;
}

export default function ModelResults() {
  const { serverPrediction } = useImageDetectionContext();

  if (!serverPrediction) {
    return (
      <div className="flex flex-col items-center">
        <h2>Make your first prediction!</h2>
      </div>
    );
  }

  const { imgData, predictedClass, cropped, error, detected } =
    serverPrediction;

  return (
    <div className="flex flex-col items-center">
      <h2>Results</h2>
      {imgData && (
        <Image
          src={imgData}
          alt="model prediction"
          width={500}
          height={384}
          className="max-w-full"
        />
      )}
      {predictedClass && <p>Predicted class: {predictedClass}</p>}
      {detected && (
        <div>
          Image detected by YOLO:
          <Image
            src={detected}
            alt="cropped image"
            width={500}
            height={384}
            className="max-w-full"
          />
        </div>
      )}
      {cropped && (
        <div>
          Image cropped after detection:
          <Image
            src={cropped}
            alt="cropped image"
            width={500}
            height={384}
            className="max-w-full"
          />
        </div>
      )}
      {error && <p className="text-orange-400">Error: {error}</p>}
    </div>
  );
}
