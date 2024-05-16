"use client";

import React from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { ModelType } from "@/enums/model-type.enum";
import { useImageDetectionContext } from "@/providers/image-detection-provider";
import { Slider } from "../ui/slider";

const formSchema = z.object({
  chosenModel: z.enum(Object.values(ModelType) as [string, ...string[]], {
    required_error: "You need to select a model.",
  }),
  confidence: z.number().min(0).max(1),
});

export default function ModelConfigForm() {
  const { setChosenModel, chosenModel, setConfidence, confidence } =
    useImageDetectionContext();
  const [sliderValue, setSliderValue] = React.useState(confidence || 0.5);

  const modelConfigurationForm = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      chosenModel: chosenModel || ModelType.KOG,
      confidence: confidence || 0.5,
    },
  });

  function onSubmit(data: z.infer<typeof formSchema>) {
    setChosenModel(data.chosenModel as ModelType);
    setConfidence(data.confidence);
    setSliderValue(data.confidence);
  }

  return (
    <div className="flex flex-col items-start w-full">
      <Form {...modelConfigurationForm}>
        <form onSubmit={modelConfigurationForm.handleSubmit(onSubmit)}>
          <FormField
            control={modelConfigurationForm.control}
            name="chosenModel"
            render={({ field }) => (
              <FormItem className="space-y-3">
                <FormLabel>
                  <h3>What model to use?</h3>
                </FormLabel>
                <FormControl>
                  <RadioGroup
                    onValueChange={(value) => {
                      field.onChange(value);
                      onSubmit({
                        ...modelConfigurationForm.getValues(),
                        chosenModel: value,
                      });
                    }}
                    defaultValue={field.value}
                    className="flex flex-col space-y-1"
                  >
                    <FormItem className="flex items-center space-x-3 space-y-0">
                      <FormControl>
                        <RadioGroupItem value={ModelType.KOG} />
                      </FormControl>
                      <FormLabel className="font-normal">KOG</FormLabel>
                    </FormItem>
                    <FormItem className="flex items-center space-x-3 space-y-0">
                      <FormControl>
                        <RadioGroupItem value={ModelType.YOLO} />
                      </FormControl>
                      <FormLabel className="font-normal">YOLO</FormLabel>
                    </FormItem>
                    <FormItem className="flex items-center space-x-3 space-y-0">
                      <FormControl>
                        <RadioGroupItem value={ModelType.KOG_YOLO} />
                      </FormControl>
                      <FormLabel className="font-normal">KOG + YOLO</FormLabel>
                    </FormItem>
                  </RadioGroup>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={modelConfigurationForm.control}
            name="confidence"
            render={({ field }) => (
              <FormItem className="space-y-3">
                <FormLabel>
                  <h3>Confidence Level: {sliderValue.toFixed(2)}</h3>
                </FormLabel>
                <FormControl>
                  <Slider
                    defaultValue={[sliderValue]}
                    max={1}
                    step={0.01}
                    className="w-[60%]"
                    onValueChange={(value) => {
                      field.onChange(value[0]);
                      modelConfigurationForm.setValue("confidence", value[0]);
                      setConfidence(value[0]);
                      setSliderValue(value[0]);
                      onSubmit({
                        ...modelConfigurationForm.getValues(),
                        confidence: value[0],
                      });
                    }}
                  />
                </FormControl>
              </FormItem>
            )}
          />
        </form>
      </Form>
    </div>
  );
}
