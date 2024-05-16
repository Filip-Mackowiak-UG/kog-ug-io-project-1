import ModelConfig from "@/components/model/model-config";
import ModelResults from "@/components/model/model-results";
import Camera from "@/components/shared/Camera";
import ImageDetectionContextProvider from "@/providers/image-detection-provider";

export default function Home() {
  return (
    <div className="grid grid-cols-3 my-4">
      <ImageDetectionContextProvider>
        <ModelConfig />
        <div className="flex flex-col gap-8">
          <Camera />
        </div>
        <ModelResults />
      </ImageDetectionContextProvider>
    </div>
  );
}
