import { useRouter } from "expo-router";
import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert, ActivityIndicator } from "react-native";

import { authClient } from "@/lib/auth-client";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    const { error } = await authClient.signIn.email({ email, password });
    
    if (error) {
      Alert.alert("Login Failed", "Invalid email or password. Please try again.");
    } else {
      router.back();
      router.replace("/profile");
    }
    setLoading(false);
  };

  return (
    <View className="flex-1 bg-gray-100 p-6 justify-center">
      <Text className="text-3xl font-bold text-center mb-8 text-gray-800">
        Welcome Back
      </Text>
      <View className="space-y-4">
        <TextInput
          className="bg-white border border-gray-300 rounded-lg px-4 py-3 text-base"
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        <TextInput
          className="bg-white border border-gray-300 rounded-lg px-4 py-3 text-base"
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />
        <TouchableOpacity
          onPress={handleLogin}
          className="bg-blue-600 rounded-lg p-4 items-center justify-center"
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text className="text-white font-bold text-base">Login</Text>
          )}
        </TouchableOpacity>
      </View>
      <TouchableOpacity
        onPress={() => router.replace("/(auth)/signup")}
        className="mt-6"
      >
        <Text className="text-center text-blue-600">
          Don't have an account? Sign Up
        </Text>
      </TouchableOpacity>
    </View>
  );
}