import React from "react";
import ModelConfigForm from "./model-config-form";

export default function ModelConfig() {
  return (
    <div className="flex flex-col items-center px-2">
      <h2 className="mb-4">Configuration</h2>
      <div className="flex items-start w-full">
        <ModelConfigForm />
      </div>
    </div>
  );
}
