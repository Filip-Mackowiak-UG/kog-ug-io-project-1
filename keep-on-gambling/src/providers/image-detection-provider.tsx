"use client";

import { ModelType } from "@/enums/model-type.enum";
import { createContext, useContext, useState } from "react";

interface ImageDetectionContextProviderProps {
  children: React.ReactNode;
}

type Img = string | null;
type Prediction = any | null;

interface ImageDetectionContext {
  processedImg: Img;
  setProcessedImg: React.Dispatch<React.SetStateAction<Img>>;
  serverPrediction: Prediction;
  setServerPrediction: React.Dispatch<React.SetStateAction<Prediction>>;
  chosenModel: ModelType | null;
  setChosenModel: React.Dispatch<React.SetStateAction<ModelType | null>>;
  confidence: number | null;
  setConfidence: React.Dispatch<React.SetStateAction<number | null>>;
}

const ImageDetectionContext = createContext<ImageDetectionContext | null>(null);

export default function ImageDetectionContextProvider({
  children,
}: ImageDetectionContextProviderProps): JSX.Element {
  const [processedImg, setProcessedImg] = useState<string | null>(null);
  const [serverPrediction, setServerPrediction] = useState<Prediction>(null);
  const [chosenModel, setChosenModel] = useState<ModelType | null>(
    ModelType.KOG
  );
  const [confidence, setConfidence] = useState<number | null>(0.5);

  return (
    <ImageDetectionContext.Provider
      value={{
        processedImg,
        setProcessedImg,
        serverPrediction,
        setServerPrediction,
        chosenModel,
        setChosenModel,
        confidence,
        setConfidence,
      }}
    >
      {children}
    </ImageDetectionContext.Provider>
  );
}

export function useImageDetectionContext(): ImageDetectionContext {
  const context = useContext(ImageDetectionContext);
  if (!context)
    throw new Error(
      "useImageDetectionContext must be used within a ImageDetectionProvider"
    );
  return context;
}
