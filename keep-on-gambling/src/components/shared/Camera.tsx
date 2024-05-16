"use client";

import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useImageDetectionContext } from "@/providers/image-detection-provider";
import { ModelType } from "@/enums/model-type.enum";

const endpointMap = {
  [ModelType.KOG]: "/detect/kog",
  [ModelType.YOLO]: "/detect/yolo",
  [ModelType.KOG_YOLO]: "/detect/yolo-kog",
};

export default function Camera() {
  const { setServerPrediction, chosenModel, confidence } =
    useImageDetectionContext();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraStarted, setCameraStarted] = useState<boolean>(false);

  useEffect(() => {
    if (cameraStarted) {
      startCamera();
    }
  }, [cameraStarted]);

  const startCamera = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError("No camera available.");
        setCameraStarted(false);
      }
    } else {
      setError("No camera available.");
      setCameraStarted(false);
    }
  };

  const takeScreenshot = () => {
    if (canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      // Set the canvas dimensions to match the video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw the current frame of the video onto the canvas
      canvas
        .getContext("2d")
        ?.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

      // Convert the canvas image to a data URL
      const imgData = canvas.toDataURL("image/png");

      // Create a link and download the image
      const link = document.createElement("a");
      link.href = imgData;
      link.download = "screenshot.png";
      link.click();
    }
  };

  const checkImage = async () => {
    if (canvasRef.current && videoRef.current && chosenModel) {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      // Set the canvas dimensions to match the video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw the current frame of the video onto the canvas
      canvas
        .getContext("2d")
        ?.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

      // Convert the canvas image to a data URL
      const imgData = canvas.toDataURL("image/png");

      // Send the image data to the server
      const response = await fetch(
        `http://localhost:3005${endpointMap[chosenModel]}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ imgData, confidence }), // Add confidence here
        }
      );

      const data = await response.json();
      setServerPrediction(data);
      console.log(data);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="border border-solid border-slate-500 rounded-sm w-[40rem] h-[30rem] grid place-items-center">
        {!cameraStarted ? (
          <div className="flex flex-col gap-2 w-1/2">
            <Button onClick={() => setCameraStarted(true)} className="">
              Start Camera
            </Button>
            {error && <p className="text-center text-red-400">{error}</p>}
          </div>
        ) : (
          <video ref={videoRef} autoPlay className="max-h-full"></video>
        )}
      </div>
      <div className="flex gap-4">
        <Button onClick={takeScreenshot}>Take screenshot</Button>
        <Button onClick={checkImage}>Check my card</Button>
      </div>
      <canvas ref={canvasRef} className="hidden"></canvas>
    </div>
  );
}
