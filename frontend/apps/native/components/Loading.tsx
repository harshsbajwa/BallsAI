import React from "react";
import { View, Text, ActivityIndicator } from "react-native";

interface LoadingProps {
  text?: string;
  size?: "small" | "large";
}

export function Loading({
  text = "Loading...",
  size = "small",
}: LoadingProps) {
  return (
    <View className="flex-row items-center justify-center space-x-2 p-4">
      <ActivityIndicator size={size} color="#3B82F6" />
      <Text className="text-gray-500 ml-2">{text}</Text>
    </View>
  );
}